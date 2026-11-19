using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace SupplyGuard.Security;

public enum EventType
{
    Order,
    ShipmentUpdate,
    Disruption,
    SecurityEvent
}

public static class EventTypeMapper
{
    public static EventType FromDb(string v) => v.ToLowerInvariant() switch
    {
        "order" => EventType.Order,
        "shipment_update" => EventType.ShipmentUpdate,
        "disruption" => EventType.Disruption,
        "security_event" => EventType.SecurityEvent,
        _ => EventType.SecurityEvent
    };
}

public sealed class SupplyNode
{
    public required string NodeId { get; init; }
    public required string Name { get; init; }
    public required string NodeType { get; init; }
    public required int Tier { get; init; }
    public required double CapacityPerTick { get; init; }
    public double TrustScore { get; set; } = 0.95;
    public bool IsCompromised { get; set; }
    public string MetadataJson { get; set; } = "{}";
}

public sealed class SupplyMessageRecord
{
    public required string MessageId { get; init; }
    public required int Tick { get; init; }
    public required string SenderId { get; set; }
    public required string RecipientId { get; init; }
    public required EventType EventType { get; set; }
    public required Dictionary<string, object?> Payload { get; set; }
    public string? Signature { get; set; }
    public bool Verified { get; set; } = true;
    public string? SecurityAlert { get; set; }
}

public sealed class NodeCredential : IDisposable
{
    public required string NodeId { get; init; }
    public required HashSet<EventType> AllowedEvents { get; init; }
    public required RSA KeyPair { get; init; }
    public required byte[] SessionAesKey { get; init; }

    public void Dispose() => KeyPair.Dispose();
}

public sealed record AuditResult(
    List<SupplyNode> UpdatedNodes,
    List<SupplyMessageRecord> AuditedMessages,
    Dictionary<string, (double TrustScore, int AlertCount, bool IsCompromised)> Summary
);

public sealed class SecurityVerifier : IDisposable
{
    private readonly double _compromiseThreshold;
    public Dictionary<string, NodeCredential> Credentials { get; } = new();

    public SecurityVerifier(IEnumerable<SupplyNode> nodes, double compromiseThreshold = 0.60)
    {
        _compromiseThreshold = compromiseThreshold;
        foreach (var n in nodes)
        {
            HashSet<EventType> allowed = n.Tier switch
            {
                0 or 1 => [EventType.ShipmentUpdate, EventType.Disruption, EventType.SecurityEvent],
                2 => [EventType.Order, EventType.Disruption, EventType.SecurityEvent],
                _ => [EventType.Order, EventType.ShipmentUpdate, EventType.SecurityEvent]
            };

            Credentials[n.NodeId] = new NodeCredential
            {
                NodeId = n.NodeId,
                AllowedEvents = allowed,
                KeyPair = CryptoPrimitives.GenerateRsaKeyPair(2048),
                SessionAesKey = CryptoPrimitives.GenerateAes128Key()
            };
        }
    }

    public SupplyMessageRecord SignMessage(SupplyMessageRecord msg)
    {
        if (!Credentials.TryGetValue(msg.SenderId, out var cred))
        {
            msg.Signature = null;
            return msg;
        }
        msg.Signature = CryptoPrimitives.SignPayload(cred.KeyPair, msg.Payload);
        return msg;
    }

    public (SupplyMessageRecord Msg, double Trust, int Alerts) VerifyMessage(
        SupplyMessageRecord msg,
        Dictionary<string, double> trustScores,
        Dictionary<string, int> alertCounts)
    {
        double trust = trustScores.GetValueOrDefault(msg.SenderId, 0.85);
        int alerts = alertCounts.GetValueOrDefault(msg.SenderId, 0);

        if (!Credentials.TryGetValue(msg.SenderId, out var cred))
        {
            msg.Verified = false;
            msg.SecurityAlert = "unknown_sender";
            return (msg, Math.Max(0.05, Math.Round(trust - 0.15, 4)), alerts + 1);
        }

        if (!cred.AllowedEvents.Contains(msg.EventType))
        {
            msg.Verified = false;
            msg.SecurityAlert = "rbac_unauthorized_event";
            return (msg, Math.Max(0.05, Math.Round(trust - 0.12, 4)), alerts + 1);
        }

        if (string.IsNullOrWhiteSpace(msg.Signature))
        {
            msg.Verified = false;
            msg.SecurityAlert = "missing_signature";
            return (msg, Math.Max(0.05, Math.Round(trust - 0.10, 4)), alerts + 1);
        }

        if (!CryptoPrimitives.VerifyPayloadSignature(cred.KeyPair, msg.Payload, msg.Signature))
        {
            msg.Verified = false;
            msg.SecurityAlert = "invalid_signature_or_tampered_payload";
            return (msg, Math.Max(0.05, Math.Round(trust - 0.15, 4)), alerts + 1);
        }

        msg.Verified = true;
        msg.SecurityAlert = null;
        return (msg, Math.Min(1.0, Math.Round(trust + 0.004, 4)), alerts);
    }

    public AuditResult AuditMessages(
        IEnumerable<SupplyNode> nodes,
        IEnumerable<SupplyMessageRecord> messages,
        bool resignClean = true)
    {
        var nodeList = nodes.ToList();
        var trustScores = nodeList.ToDictionary(n => n.NodeId, n => n.TrustScore);
        var alertCounts = nodeList.ToDictionary(n => n.NodeId, _ => 0);
        var audited = new List<SupplyMessageRecord>();

        foreach (var m in messages.OrderBy(x => x.Tick).ThenBy(x => x.MessageId, StringComparer.Ordinal))
        {
            var copy = new SupplyMessageRecord
            {
                MessageId = m.MessageId,
                Tick = m.Tick,
                SenderId = m.SenderId,
                RecipientId = m.RecipientId,
                EventType = m.EventType,
                Payload = new Dictionary<string, object?>(m.Payload),
                Signature = m.Signature,
                Verified = m.Verified,
                SecurityAlert = m.SecurityAlert
            };

            if (resignClean && copy.Verified && copy.SecurityAlert is null && Credentials.ContainsKey(copy.SenderId))
                SignMessage(copy);

            var (vMsg, nextTrust, nextAlerts) = VerifyMessage(copy, trustScores, alertCounts);
            if (trustScores.ContainsKey(vMsg.SenderId))
            {
                trustScores[vMsg.SenderId] = nextTrust;
                alertCounts[vMsg.SenderId] = nextAlerts;
            }
            audited.Add(vMsg);
        }

        var updatedNodes = new List<SupplyNode>();
        var summary = new Dictionary<string, (double, int, bool)>();

        foreach (var n in nodeList)
        {
            double t = trustScores.GetValueOrDefault(n.NodeId, n.TrustScore);
            int a = alertCounts.GetValueOrDefault(n.NodeId, 0);
            bool comp = t < _compromiseThreshold || a >= 4;

            var meta = string.IsNullOrWhiteSpace(n.MetadataJson)
                ? new Dictionary<string, object?>()
                : JsonSerializer.Deserialize<Dictionary<string, object?>>(n.MetadataJson) ?? new();
            meta["security_alerts_count"] = a;

            updatedNodes.Add(new SupplyNode
            {
                NodeId = n.NodeId,
                Name = n.Name,
                NodeType = n.NodeType,
                Tier = n.Tier,
                CapacityPerTick = n.CapacityPerTick,
                TrustScore = t,
                IsCompromised = comp,
                MetadataJson = JsonSerializer.Serialize(meta)
            });
            summary[n.NodeId] = (t, a, comp);
        }

        return new AuditResult(updatedNodes, audited, summary);
    }

    public static void PersistToSqlite(
        string dbPath,
        IEnumerable<SupplyNode> updatedNodes,
        IEnumerable<SupplyMessageRecord> auditedMessages)
    {
        using var conn = new SqliteConnection($"Data Source={dbPath}");
        conn.Open();
        using var tx = conn.BeginTransaction();

        using (var cmd = conn.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "UPDATE nodes SET trust_score = $t, is_compromised = $c, metadata_json = $m WHERE node_id = $id;";
            var pT = cmd.Parameters.Add("$t", SqliteType.Real);
            var pC = cmd.Parameters.Add("$c", SqliteType.Integer);
            var pM = cmd.Parameters.Add("$m", SqliteType.Text);
            var pId = cmd.Parameters.Add("$id", SqliteType.Text);

            foreach (var n in updatedNodes)
            {
                pT.Value = n.TrustScore;
                pC.Value = n.IsCompromised ? 1 : 0;
                pM.Value = n.MetadataJson;
                pId.Value = n.NodeId;
                cmd.ExecuteNonQuery();
            }
        }

        using (var cmd = conn.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "UPDATE supply_messages SET sender_id = $s, signature = $sig, verified = $v, security_alert = $a WHERE message_id = $mid;";
            var pS = cmd.Parameters.Add("$s", SqliteType.Text);
            var pSig = cmd.Parameters.Add("$sig", SqliteType.Text);
            var pV = cmd.Parameters.Add("$v", SqliteType.Integer);
            var pA = cmd.Parameters.Add("$a", SqliteType.Text);
            var pMid = cmd.Parameters.Add("$mid", SqliteType.Text);

            foreach (var m in auditedMessages)
            {
                pS.Value = m.SenderId;
                pSig.Value = (object?)m.Signature ?? DBNull.Value;
                pV.Value = m.Verified ? 1 : 0;
                pA.Value = (object?)m.SecurityAlert ?? DBNull.Value;
                pMid.Value = m.MessageId;
                cmd.ExecuteNonQuery();
            }
        }

        tx.Commit();
    }

    public void Dispose()
    {
        foreach (var c in Credentials.Values) c.Dispose();
    }
}