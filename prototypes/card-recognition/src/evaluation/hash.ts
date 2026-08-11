export async function sha256Hex(blob: Blob): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function normalizeSha256(value: string): string {
  const normalized = value.toLowerCase().replace(/^sha256:/, "");

  if (!/^[a-f0-9]{64}$/.test(normalized)) {
    throw new TypeError("SHA-256 invalide dans le manifeste");
  }

  return normalized;
}
