import type { StylePresetRecordWithImage } from "services/api/endpoints/stylePresets";
import { StylePresetFormData } from "../components/StylePresetForm";

export type StylePresetModalState = {
    isModalOpen: boolean;
    updatingStylePresetId: string | null;
    prefilledFormData: StylePresetFormData | null
};


export type StylePresetState = {
    isMenuOpen: boolean;
    activeStylePreset: StylePresetRecordWithImage | null;
    searchTerm: string;
    viewMode: boolean;
}

