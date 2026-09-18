"use client";

import { ArrowsOut, Info } from "@phosphor-icons/react";
import {
  Background,
  BackgroundVariant,
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  Handle,
  MiniMap,
  Panel,
  ReactFlow,
  getSmoothStepPath,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
  Position,
} from "@xyflow/react";
import dagre from "dagre";
import { useMemo } from "react";
import "@xyflow/react/dist/style.css";
import type { GraphNode, GraphEdge } from "@/lib/types";

interface PaperGraphData extends Record<string, unknown> {
  node: GraphNode;
}

type PaperFlowNode = Node<PaperGraphData, "paperNode">;

function PaperNode({ data }: NodeProps<PaperFlowNode>) {
  const node = data.node;
  const isAccent = node.type === "proposed" || node.type === "decision";
  return (
    <div className={`min-w-[176px] rounded-[12px] border px-3 py-3 shadow-[0_8px_24px_rgba(31,46,40,0.06)] ${isAccent ? "border-[var(--accent)] bg-[var(--accent-soft)]" : "border-[var(--line)] bg-[var(--surface)]"}`}>
      <Handle type="target" position={Position.Left} className="!h-2 !w-2 !border-[var(--surface)] !bg-[var(--accent)]" />
      <Handle type="source" position={Position.Right} className="!h-2 !w-2 !border-[var(--surface)] !bg-[var(--accent)]" />
      <div className="mb-2 flex items-center justify-between gap-3">
        <span className="mono-font text-[9px] uppercase tracking-[0.16em] text-[var(--ink-faint)]">{node.type}</span>
        <Info size={13} className="text-[var(--ink-faint)]" />
      </div>
      <p className="text-[12px] font-semibold leading-4 text-[var(--ink)]">{node.label}</p>
      <p className="mt-1.5 max-w-[150px] text-[10px] leading-4 text-[var(--ink-muted)]">{node.explanation}</p>
    </div>
  );
}

const nodeTypes = { paperNode: PaperNode };

function PaperEdge({
  id,
  sourceX,
  sourceY,
  sourcePosition,
  targetX,
  targetY,
  targetPosition,
  label,
  style,
  markerEnd,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 12,
    offset: 22,
  });

  return (
    <>
      <BaseEdge id={id} path={edgePath} style={style} markerEnd={markerEnd} />
      {label && (
        <EdgeLabelRenderer>
          <div
            className="nodrag nopan pointer-events-none max-w-[190px] break-words rounded-full border border-[var(--line)] bg-[var(--surface)] px-2 py-1 text-center font-mono text-[9px] leading-3 text-[var(--ink-muted)] shadow-[0_2px_10px_rgba(31,46,40,0.12)]"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: "none",
              zIndex: 20,
            }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

const edgeTypes = { paperEdge: PaperEdge };

function getLayoutedElements(graphNodes: GraphNode[], graphEdges: GraphEdge[]) {
  const graph = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));
  const longestLabel = graphEdges.reduce((length, edge) => Math.max(length, (edge.label ?? edge.relation ?? "").length), 0);
  const ranksep = Math.max(130, Math.min(220, 64 + longestLabel * 3.5));
  graph.setGraph({ rankdir: "LR", ranksep, nodesep: 42, marginx: 20, marginy: 20 });

  graphNodes.forEach((node) => graph.setNode(node.id, { width: 176, height: 150 }));
  graphEdges.forEach((edge) => graph.setEdge(edge.source, edge.target));
  dagre.layout(graph);

  const nodes: PaperFlowNode[] = graphNodes.map((node) => {
    const position = graph.node(node.id);
    return {
      id: node.id,
      type: "paperNode",
      position: { x: position.x - 88, y: position.y - 75 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      data: { node },
    };
  });
  const edges: Edge[] = graphEdges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.label ?? edge.relation,
    type: "paperEdge",
    style: { stroke: "var(--line-strong)", strokeWidth: 1.4 },
  }));
  return { nodes, edges };
}

export function ConceptGraph({ graph, onEvidence }: { graph: { nodes: GraphNode[]; edges: GraphEdge[] }; onEvidence?: (evidenceIds: string[]) => void }) {
  const layout = useMemo(() => getLayoutedElements(graph.nodes, graph.edges), [graph]);

  return (
    <div className="relative h-[520px] overflow-hidden rounded-[16px] border border-[var(--line)] bg-[var(--surface)]" aria-label="Interactive paper concept graph">
      <ReactFlow
        nodes={layout.nodes}
        edges={layout.edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={false}
        nodesConnectable={false}
        onNodeClick={(_, node) => onEvidence?.(node.data.node.evidenceIds)}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--line)" />
        <Controls showInteractive={false} position="bottom-left" />
        <MiniMap pannable zoomable nodeColor="var(--accent)" maskColor="color-mix(in srgb, var(--canvas) 68%, transparent)" />
        <Panel position="top-left" className="!m-4 rounded-[9px] border border-[var(--line)] bg-[var(--surface)]/90 px-3 py-2 text-[10px] text-[var(--ink-muted)] shadow-sm backdrop-blur">
          <span className="flex items-center gap-2"><ArrowsOut size={13} /> Drag to inspect the relationships</span>
        </Panel>
      </ReactFlow>
    </div>
  );
}
