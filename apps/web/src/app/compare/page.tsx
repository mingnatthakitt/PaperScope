import { CompareWorkspace } from "@/components/compare-workspace";

export default async function ComparePage({ searchParams }: { searchParams: Promise<{ papers?: string }> }) {
  const params = await searchParams;
  const paperIds = (params.papers ?? "").split(",").map((id) => id.trim()).filter(Boolean).slice(0, 3);
  return <CompareWorkspace paperIds={paperIds} />;
}
