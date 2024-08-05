import type { StylePresetRecordDTO } from "services/api/endpoints/stylePresets";

export type StylePresetModalState = {
    isModalOpen: boolean;
    updatingStylePreset: StylePresetRecordDTO | null;
};

export type StylePresetState = {
    isMenuOpen: boolean;
    activeStylePreset: StylePresetRecordDTO | null;
    calculatedPosPrompt?: string
    calculatedNegPrompt?: string
}

