import { useEffect, useState } from "react";
import axios from "axios";

export default function DocumentList({ refreshKey }) {
  const [docs, setDocs] = useState([]);

  const fetchDocs = async () => {
    const res = await axios.get("http://localhost:8000/documents/");
    setDocs(res.data);
  };

  useEffect(() => {
    fetchDocs();
  }, [refreshKey]);

  const handleDelete = async (id) => {
    await axios.delete(`http://localhost:8000/documents/${id}`);
    fetchDocs();
  };

  return (
    <div>
      {docs.length === 0 && <p style={{ color: "var(--ink-dim)", fontSize: "0.9rem" }}>No documents uploaded yet.</p>}
      {docs.map((doc) => (
        <div className="doc-row" key={doc.id}>
          <span className="mono">{doc.filename} <span style={{ color: "var(--copper)" }}>· {doc.chunks}ch</span></span>
          <button onClick={() => handleDelete(doc.id)}>×</button>
        </div>
      ))}
    </div>
  );
}