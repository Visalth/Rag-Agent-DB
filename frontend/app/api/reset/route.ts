const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: Request) {
  const { session } = await req.json();
  const upstream = await fetch(
    `${BACKEND}/reset?session=${encodeURIComponent(session ?? "default")}`,
    { method: "DELETE" }
  );
  const text = await upstream.text();
  return new Response(text, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
