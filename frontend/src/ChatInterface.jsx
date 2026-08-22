import { useState } from "react";
import axios from "axios";
import SourceCard from "./SourceCard";

export default function ChatInterface() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState(null);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleAsk = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setAnswer(null);

    try {
      const res = await axios.post("http://localhost:8000/search/qa", {
        query,
        top_k: 5,
      });
      setAnswer(res.data.answer);
      setSources(res.data.sources || []);
    } catch (err) {
      setAnswer("Something went wrong: " + (err.response?.data?.detail || err.message));
      setSources([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="What do you want to know?"
          style={{ flex: 1 }}
        />
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>

      {answer && (
        <div style={{ marginBottom: "1.5rem" }}>
          <p className="label" style={{ marginBottom: "0.5rem" }}>Answer</p>
          <p className="answer-text">{answer}</p>
        </div>
      )}

      {sources.length > 0 && (
        <div>
          <p className="label" style={{ marginBottom: "0.5rem" }}>Sources</p>
          {sources.map((s, i) => (
            <SourceCard key={s.chunk_id} source={s} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}