const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: Request) {
  const form = await req.formData();
  const upstream = await fetch(`${BACKEND}/upload`, {
    method: "POST",
    body: form,
  });
  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
