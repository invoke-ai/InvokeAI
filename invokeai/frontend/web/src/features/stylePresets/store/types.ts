import type { StylePresetRecordDTO } from "services/api/endpoints/stylePresets";

export type StylePresetState = {
    isModalOpen: boolean;
    updatingStylePreset: StylePresetRecordDTO | null;
};


