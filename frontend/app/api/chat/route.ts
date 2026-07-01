const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.json();
  const upstream = await fetch(`${BACKEND}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    return new Response(text, { status: upstream.status || 502 });
  }

  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-store",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
