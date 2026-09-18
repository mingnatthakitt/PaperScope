import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatAuthors(authors: string[]) {
  if (authors.length <= 2) return authors.join(" and ");
  return `${authors.slice(0, 2).join(", ")}, et al.`;
}

export function formatScore(score: number) {
  return `${Math.round(score * 100)}%`;
}

export function formatPageCount(pageCount: number) {
  return `${pageCount} ${pageCount === 1 ? "page" : "pages"}`;
}
