using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SupplyGuard.Security;

public static class CryptoPrimitives
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public static string CaesarEncrypt(string text, int shift = 3)
    {
        int s = ((shift % 26) + 26) % 26;
        var sb = new StringBuilder(text.Length);
        foreach (char ch in text)
        {
            if (ch is >= 'a' and <= 'z')
                sb.Append((char)('a' + ((ch - 'a' + s) % 26)));
            else if (ch is >= 'A' and <= 'Z')
                sb.Append((char)('A' + ((ch - 'A' + s) % 26)));
            else
                sb.Append(ch);
        }
        return sb.ToString();
    }

    public static string CaesarDecrypt(string text, int shift = 3) =>
        CaesarEncrypt(text, -shift);

    public static byte[] CanonicalBytes(Dictionary<string, object?> payload)
    {
        var sorted = new SortedDictionary<string, object?>(payload, StringComparer.Ordinal);
        return Encoding.UTF8.GetBytes(JsonSerializer.Serialize(sorted, JsonOpts));
    }

    public static string Sha256Hex(string text)
    {
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    public static string ComputeHmacSha256(byte[] key, Dictionary<string, object?> payload)
    {
        byte[] mac = HMACSHA256.HashData(key, CanonicalBytes(payload));
        return Convert.ToHexString(mac).ToLowerInvariant();
    }

    public static bool VerifyHmacSha256(byte[] key, Dictionary<string, object?> payload, string expectedHex)
    {
        string actual = ComputeHmacSha256(key, payload);
        return CryptographicOperations.FixedTimeEquals(
            Encoding.UTF8.GetBytes(actual),
            Encoding.UTF8.GetBytes(expectedHex.ToLowerInvariant())
        );
    }

    public static byte[] GenerateAes128Key() => RandomNumberGenerator.GetBytes(16);

    public static (string Nonce, string Ciphertext, string Tag) AesGcmEncrypt(
        byte[] key, Dictionary<string, object?> payload, byte[]? aad = null)
    {
        byte[] plaintext = CanonicalBytes(payload);
        byte[] nonce = RandomNumberGenerator.GetBytes(AesGcm.NonceByteSizes.MaxSize);
        byte[] ciphertext = new byte[plaintext.Length];
        byte[] tag = new byte[AesGcm.TagByteSizes.MaxSize];

        using var aes = new AesGcm(key, AesGcm.TagByteSizes.MaxSize);
        aes.Encrypt(nonce, plaintext, ciphertext, tag, aad);

        return (
            Convert.ToBase64String(nonce),
            Convert.ToBase64String(ciphertext),
            Convert.ToBase64String(tag)
        );
    }

    public static Dictionary<string, object?> AesGcmDecrypt(
        byte[] key, string nonceB64, string cipherB64, string tagB64, byte[]? aad = null)
    {
        byte[] nonce = Convert.FromBase64String(nonceB64);
        byte[] ciphertext = Convert.FromBase64String(cipherB64);
        byte[] tag = Convert.FromBase64String(tagB64);
        byte[] plaintext = new byte[ciphertext.Length];

        using var aes = new AesGcm(key, AesGcm.TagByteSizes.MaxSize);
        aes.Decrypt(nonce, ciphertext, tag, plaintext, aad);

        string json = Encoding.UTF8.GetString(plaintext);
        return JsonSerializer.Deserialize<Dictionary<string, object?>>(json) ?? new();
    }

    public static RSA GenerateRsaKeyPair(int bits = 2048) => RSA.Create(bits);

    public static string RsaWrapKey(RSA publicKey, byte[] symmetricKey) =>
        Convert.ToBase64String(publicKey.Encrypt(symmetricKey, RSAEncryptionPadding.OaepSHA256));

    public static byte[] RsaUnwrapKey(RSA privateKey, string wrappedB64) =>
        privateKey.Decrypt(Convert.FromBase64String(wrappedB64), RSAEncryptionPadding.OaepSHA256);

    public static string SignPayload(RSA privateKey, Dictionary<string, object?> payload)
    {
        byte[] sig = privateKey.SignData(
            CanonicalBytes(payload), HashAlgorithmName.SHA256, RSASignaturePadding.Pss
        );
        return Convert.ToBase64String(sig);
    }

    public static bool VerifyPayloadSignature(RSA publicKey, Dictionary<string, object?> payload, string? sigB64)
    {
        if (string.IsNullOrWhiteSpace(sigB64)) return false;
        try
        {
            return publicKey.VerifyData(
                CanonicalBytes(payload),
                Convert.FromBase64String(sigB64),
                HashAlgorithmName.SHA256,
                RSASignaturePadding.Pss
            );
        }
        catch
        {
            return false;
        }
    }
}