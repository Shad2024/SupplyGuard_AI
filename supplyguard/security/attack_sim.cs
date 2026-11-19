namespace SupplyGuard.Security;

public sealed record AttackSimOutcome(
    string AttackType,
    bool Verified,
    string? Alert,
    double UpdatedTrust,
    int Alerts
);

public static class AttackSimulator
{
    public static List<AttackSimOutcome> Run(SupplyMessageRecord baseMsg, SecurityVerifier verifier)
    {
        var trust = new Dictionary<string, double> { [baseMsg.SenderId] = 0.90 };
        var alerts = new Dictionary<string, int> { [baseMsg.SenderId] = 0 };
        var outcomes = new List<AttackSimOutcome>();

        var leg = Clone(baseMsg);
        verifier.SignMessage(leg);
        var (o1, t1, a1) = verifier.VerifyMessage(leg, trust, alerts);
        outcomes.Add(new("none_legitimate", o1.Verified, o1.SecurityAlert, t1, a1));

        var tamp = Clone(baseMsg);
        verifier.SignMessage(tamp);
        tamp.Payload["qty"] = 9999.0;
        var (o2, t2, a2) = verifier.VerifyMessage(tamp, trust, alerts);
        outcomes.Add(new("tampered_payload", o2.Verified, o2.SecurityAlert, t2, a2));

        var spoof = Clone(baseMsg);
        verifier.SignMessage(spoof);
        spoof.SenderId = "SPOOF-SUPPLIER-999";
        var (o3, t3, a3) = verifier.VerifyMessage(spoof, trust, alerts);
        outcomes.Add(new("spoofed_sender", o3.Verified, o3.SecurityAlert, t3, a3));

        var rbac = Clone(baseMsg);
        rbac.Payload["unauthorized_order"] = true;
        verifier.SignMessage(rbac);
        rbac.EventType = EventType.Order;
        var (o4, t4, a4) = verifier.VerifyMessage(rbac, trust, alerts);
        outcomes.Add(new("rbac_violation", o4.Verified, o4.SecurityAlert, t4, a4));

        return outcomes;
    }

    private static SupplyMessageRecord Clone(SupplyMessageRecord m) => new()
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
}