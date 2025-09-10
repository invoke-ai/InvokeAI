import { atom } from "nanostores";

export const $isOpen = atom(false);
export const $imageName = atom<string | null>(null);