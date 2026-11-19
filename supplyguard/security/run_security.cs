using System.Text.Json;
using Microsoft.Data.Sqlite;
using SupplyGuard.Security;

string dbPath = args.Length > 0
    ? Path.GetFullPath(args[0])
    : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "artifacts", "supplyguard_laptop.db"));

if (!File.Exists(dbPath))
{
    Console.Error.WriteLine($"Database not found: {dbPath}");
    return;
}

var nodes = LoadNodes(dbPath);
var messages = LoadMessages(dbPath);

using var verifier = new SecurityVerifier(nodes, compromiseThreshold: 0.60);
var audit = verifier.AuditMessages(nodes, messages, resignClean: true);
SecurityVerifier.PersistToSqlite(dbPath, audit.UpdatedNodes, audit.AuditedMessages);

int unverified = audit.AuditedMessages.Count(m => !m.Verified);
var compromised = audit.UpdatedNodes.Where(n => n.IsCompromised).Select(n => n.NodeId).ToList();

Console.WriteLine($"DB: {dbPath}");
Console.WriteLine($"Messages: {audit.AuditedMessages.Count} | Unverified: {unverified}");
Console.WriteLine($"Compromised Nodes: {string.Join(", ", compromised)}");

var sample = messages.FirstOrDefault(m => verifier.Credentials.ContainsKey(m.SenderId));
if (sample is not null)
{
    Console.WriteLine($"\nAttack sim on {sample.MessageId} ({sample.SenderId}):");
    foreach (var r in AttackSimulator.Run(sample, verifier))
    {
        Console.WriteLine($"  {r.AttackType,-18} verified={r.Verified,-5} alert={r.Alert ?? "none",-36} trust={r.UpdatedTrust:F4}");
    }
}

static List<SupplyNode> LoadNodes(string path)
{
    var list = new List<SupplyNode>();
    using var conn = new SqliteConnection($"Data Source={path}");
    conn.Open();
    using var cmd = conn.CreateCommand();
    cmd.CommandText = "SELECT node_id, name, node_type, tier, capacity_per_tick, trust_score, is_compromised, metadata_json FROM nodes ORDER BY tier, node_id;";
    using var r = cmd.ExecuteReader();
    while (r.Read())
    {
        list.Add(new SupplyNode
        {
            NodeId = r.GetString(0),
            Name = r.GetString(1),
            NodeType = r.GetString(2),
            Tier = r.GetInt32(3),
            CapacityPerTick = r.GetDouble(4),
            TrustScore = r.GetDouble(5),
            IsCompromised = r.GetInt32(6) != 0,
            MetadataJson = r.IsDBNull(7) ? "{}" : r.GetString(7)
        });
    }
    return list;
}

static List<SupplyMessageRecord> LoadMessages(string path)
{
    var list = new List<SupplyMessageRecord>();
    using var conn = new SqliteConnection($"Data Source={path}");
    conn.Open();
    using var cmd = conn.CreateCommand();
    cmd.CommandText = "SELECT message_id, tick, sender_id, recipient_id, event_type, payload_json, signature, verified, security_alert FROM supply_messages ORDER BY tick, message_id;";
    using var r = cmd.ExecuteReader();
    while (r.Read())
    {
        var payload = JsonSerializer.Deserialize<Dictionary<string, object?>>(r.IsDBNull(5) ? "{}" : r.GetString(5)) ?? new();
        list.Add(new SupplyMessageRecord
        {
            MessageId = r.GetString(0),
            Tick = r.GetInt32(1),
            SenderId = r.GetString(2),
            RecipientId = r.GetString(3),
            EventType = EventTypeMapper.FromDb(r.GetString(4)),
            Payload = payload,
            Signature = r.IsDBNull(6) ? null : r.GetString(6),
            Verified = r.GetInt32(7) != 0,
            SecurityAlert = r.IsDBNull(8) ? null : r.GetString(8)
        });
    }
    return list;
}