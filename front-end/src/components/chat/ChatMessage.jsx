export default function ChatMessage({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed
          ${
            isUser
              ? "bg-sky-600 text-white rounded-br-md"
              : "bg-gray-100 text-gray-800 rounded-bl-md"
          }`}
      >
        {!isUser && message.tools && message.tools.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1">
            {message.tools.map((tool, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 bg-amber-100 text-amber-700
                           text-xs font-medium px-2 py-0.5 rounded-full"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                {tool.tool_name.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        )}

        <div className="whitespace-pre-wrap">{message.content}</div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-200">
            <p className="text-xs text-gray-400 mb-1">Sources:</p>
            <div className="flex flex-wrap gap-1">
              {message.sources.map((source, i) => (
                <span
                  key={i}
                  className="text-xs bg-gray-200 text-gray-500 px-2 py-0.5 rounded-full"
                >
                  {source.document}
                </span>
              ))}
            </div>
          </div>
        )}

        {!isUser && message.latency && (
          <p className="text-xs text-gray-400 mt-1">
            {(message.latency / 1000).toFixed(1)}s
          </p>
        )}
      </div>
    </div>
  );
}
