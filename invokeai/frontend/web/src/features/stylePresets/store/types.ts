import type { StylePresetRecordWithImage } from "services/api/endpoints/stylePresets";
import { ImageDTO } from "../../../services/api/types";

export type StylePresetModalState = {
    isModalOpen: boolean;
    updatingStylePreset: StylePresetRecordWithImage | null;
    createPresetFromImage: ImageDTO | null
};

export type StylePresetPrefillOptions = {
    positivePrompt: string;
    negativePrompt: string;
    image: File;
}

export type StylePresetState = {
    isMenuOpen: boolean;
    activeStylePreset: StylePresetRecordWithImage | null;
    searchTerm: string
}

