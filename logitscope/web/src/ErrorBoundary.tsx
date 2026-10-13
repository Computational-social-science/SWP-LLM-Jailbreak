// catch render crashes and show them instead of a blank screen.

import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            padding: 24,
            color: "#ff5d5d",
            fontFamily: "ui-monospace, Menlo, monospace",
            fontSize: 13,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            background: "#0a0e0c",
            height: "100%",
            overflow: "auto",
          }}
        >
          <div style={{ fontSize: 16, marginBottom: 12, fontWeight: 700 }}>
            something crashed:
          </div>
          <div>{String(this.state.error?.message ?? this.state.error)}</div>
          <div style={{ marginTop: 12, color: "#c9e4d4" }}>{this.state.error?.stack}</div>
        </div>
      );
    }
    return this.props.children;
  }
}
