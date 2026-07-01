"use client";

import { useEffect, useRef, useState } from "react";

type Msg = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  streaming?: boolean;
};

type UploadedFile = { name: string; chunks: number };

const ACCEPT = ".pdf,.docx,.csv,.txt,.md";

export default function Home() {
  const [session, setSession] = useState("");
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let s = localStorage.getItem("docchat_session");
    if (!s) {
      s = crypto.randomUUID();
      localStorage.setItem("docchat_session", s);
    }
    // one-time client init from localStorage; must run after mount
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSession(s);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleFiles(list: FileList | null) {
    if (!list || !session) return;
    setError("");
    setUploading(true);
    for (const file of Array.from(list)) {
      const form = new FormData();
      form.append("file", file);
      form.append("session", session);
      try {
        const res = await fetch("/api/upload", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) {
          setError(data.detail || `Couldn't read ${file.name}`);
          continue;
        }
        setFiles((f) => [
          ...f.filter((x) => x.name !== data.file),
          { name: data.file, chunks: data.chunks },
        ]);
      } catch {
        setError(`Upload failed for ${file.name}`);
      }
    }
    setUploading(false);
  }

  async function clearAll() {
    if (!session) return;
    await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session }),
    });
    setFiles([]);
    setMessages([]);
    setError("");
  }

  async function send() {
    const question = input.trim();
    if (!question || sending) return;
    setInput("");
    setError("");
    setSending(true);

    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "", sources: [], streaming: true },
    ]);

    const updateLast = (fn: (m: Msg) => Msg) =>
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = fn(copy[copy.length - 1]);
        return copy;
      });

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, session }),
        cache: "no-store",
      });
      if (!res.ok || !res.body) throw new Error("stream failed");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          const evt = JSON.parse(line.slice(5).trim());
          if (evt.type === "sources") {
            updateLast((m) => ({ ...m, sources: evt.sources }));
          } else if (evt.type === "token") {
            updateLast((m) => ({ ...m, content: m.content + evt.text }));
          }
        }
      }
    } catch {
      updateLast((m) => ({
        ...m,
        content: m.content || "Something went wrong. Try again.",
      }));
    } finally {
      updateLast((m) => ({
        ...m,
        content: m.content || "No response came back. Try asking again.",
        streaming: false,
      }));
      setSending(false);
    }
  }

  const empty = messages.length === 0;

  return (
    <main className="mx-auto flex h-dvh w-full max-w-2xl flex-col px-4">
      {/* Header */}
      <header className="flex items-baseline justify-between border-b border-zinc-100 py-5">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-900">
            Rag agent
          </h1>
          <p className="text-sm text-zinc-500">Answers only from your documents.</p>
        </div>
        {files.length > 0 && (
          <button
            onClick={clearAll}
            className="text-sm text-zinc-400 underline-offset-4 hover:text-zinc-900 hover:underline"
          >
            Clear all
          </button>
        )}
      </header>

      {/* Uploaded files */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 py-3">
          {files.map((f) => (
            <span
              key={f.name}
              className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs text-zinc-600"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              {f.name}
            </span>
          ))}
        </div>
      )}

      {/* Thread / empty state */}
      <div className="flex-1 overflow-y-auto py-4">
        {empty ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <UploadZone
              dragOver={dragOver}
              setDragOver={setDragOver}
              uploading={uploading}
              onFiles={handleFiles}
              large
            />
            <p className="mt-4 max-w-sm text-sm text-zinc-500">
              Drop a PDF, Word doc, CSV, or text file, then ask a question. Answers
              come only from what you upload.
            </p>
          </div>
        ) : (
          <div className="space-y-5">
            {messages.map((m, i) => (
              <Bubble key={i} msg={m} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {error && <p className="pb-2 text-sm text-red-600">{error}</p>}

      {/* Input bar */}
      <div className="border-t border-zinc-100 py-4">
        <div className="flex items-end gap-2">
          {!empty && (
            <UploadZone
              dragOver={dragOver}
              setDragOver={setDragOver}
              uploading={uploading}
              onFiles={handleFiles}
            />
          )}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={1}
            placeholder={
              files.length ? "Ask about your documents..." : "Upload a document first"
            }
            className="max-h-40 flex-1 resize-none rounded-2xl border border-zinc-200 px-4 py-3 text-sm text-zinc-900 outline-none placeholder:text-zinc-400 focus:border-zinc-400"
          />
          <button
            onClick={send}
            disabled={sending || !input.trim()}
            className="rounded-2xl bg-zinc-900 px-4 py-3 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:bg-zinc-300"
          >
            {sending ? "..." : "Send"}
          </button>
        </div>
      </div>
    </main>
  );
}

function Bubble({ msg }: { msg: Msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={isUser ? "flex justify-end" : "flex justify-start"}>
      <div className={isUser ? "max-w-[85%]" : "max-w-[90%]"}>
        <div
          className={
            isUser
              ? "rounded-2xl rounded-br-sm bg-zinc-900 px-4 py-2.5 text-sm text-white"
              : "rounded-2xl rounded-bl-sm border border-zinc-200 bg-zinc-50 px-4 py-2.5 text-sm text-zinc-900"
          }
        >
          <span className="whitespace-pre-wrap">{msg.content}</span>
          {msg.streaming && !msg.content && (
            <span className="inline-flex gap-1 align-middle">
              <Dot /> <Dot /> <Dot />
            </span>
          )}
        </div>
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <p className="mt-1.5 pl-1 text-xs text-zinc-400">
            Sources: {msg.sources.join(", ")}
          </p>
        )}
      </div>
    </div>
  );
}

function Dot() {
  return (
    <span className="inline-block h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400 [animation-duration:1s]" />
  );
}

function UploadZone({
  dragOver,
  setDragOver,
  uploading,
  onFiles,
  large = false,
}: {
  dragOver: boolean;
  setDragOver: (v: boolean) => void;
  uploading: boolean;
  onFiles: (l: FileList | null) => void;
  large?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  if (large) {
    return (
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          onFiles(e.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed px-10 py-12 transition ${dragOver
            ? "border-zinc-900 bg-zinc-50"
            : "border-zinc-200 hover:border-zinc-300"
          }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          className="hidden"
          onChange={(e) => onFiles(e.target.files)}
        />
        <span className="text-sm font-medium text-zinc-900">
          {uploading ? "Uploading..." : "Upload a document"}
        </span>
        <span className="mt-1 text-xs text-zinc-400">PDF, DOCX, CSV, TXT, MD</span>
      </label>
    );
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        className="hidden"
        onChange={(e) => onFiles(e.target.files)}
      />
      <button
        onClick={() => inputRef.current?.click()}
        title="Upload a document"
        className="rounded-2xl border border-zinc-200 px-3 py-3 text-sm text-zinc-500 transition hover:border-zinc-400 hover:text-zinc-900"
      >
        {uploading ? "..." : "+"}
      </button>
    </>
  );
}
