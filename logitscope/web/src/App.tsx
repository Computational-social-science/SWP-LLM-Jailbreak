import { useEffect } from "react";
import { init } from "./actions";
import AttentionView from "./components/AttentionView";
import BranchTree from "./components/BranchTree";
import CompareAttention from "./components/CompareAttention";
import CompareView from "./components/CompareView";
import DetailPanel from "./components/DetailPanel";
import Hud from "./components/Hud";
import PromptBar from "./components/PromptBar";
import Sparkline from "./components/Sparkline";
import StreamView from "./components/StreamView";
import Waterfall from "./components/Waterfall";
import { store, useStore } from "./store";

export default function App() {
  useEffect(() => {
    void init();
  }, []);

  const tab = useStore((s) => s.bottomTab);
  const compareMode = useStore((s) => s.compareMode);

  if (compareMode) {
    return (
      <div className="app">
        <Hud />
        <CompareAttention />
      </div>
    );
  }

  return (
    <div className="app">
      <Hud />
      <PromptBar />
      <div className="cols">
        <BranchTree />
        <StreamView />
        <DetailPanel />
      </div>
      <Sparkline />
      <div className="bottom panel">
        <div className="panel-title tabs">
          <button
            className={tab === "waterfall" ? "tab tab-on" : "tab"}
            onClick={() => store.set({ bottomTab: "waterfall" })}
          >
            logit lens waterfall
          </button>
          <button
            className={tab === "attention" ? "tab tab-on" : "tab"}
            onClick={() => store.set({ bottomTab: "attention" })}
          >
            attention
          </button>
          <button
            className={tab === "diff" ? "tab tab-on" : "tab"}
            onClick={() => store.set({ bottomTab: "diff" })}
          >
            fork diff
          </button>
        </div>
        {tab === "waterfall" ? <Waterfall /> : tab === "attention" ? <AttentionView /> : <CompareView />}
      </div>
    </div>
  );
}
