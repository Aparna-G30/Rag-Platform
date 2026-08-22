import { useState } from "react";
import DocumentUpload from "./DocumentUpload";
import DocumentList from "./DocumentList";
import ChatInterface from "./ChatInterface";

export default function App() {
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "2.5rem 2rem" }}>
      <div className="title-block">
        <h1>RAG READER</h1>
        <div className="meta">
          SHEET 01 &nbsp;·&nbsp; DOC-QA SYSTEM<br />
          REV: LIVE
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: "1.5rem" }}>
        <div>
          <div className="panel">
            <p className="label">Upload Document</p>
            <DocumentUpload onUploadComplete={() => setRefreshKey((k) => k + 1)} />
          </div>
          <div className="panel">
            <p className="label">Documents</p>
            <DocumentList refreshKey={refreshKey} />
          </div>
        </div>
        <div className="panel">
          <p className="label">Ask a Question</p>
          <ChatInterface />
        </div>
      </div>
    </div>
  );
}