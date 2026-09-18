import { PaperWorkspace } from "@/components/paper-workspace";

export default async function PaperPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <PaperWorkspace paperId={id} />;
}
