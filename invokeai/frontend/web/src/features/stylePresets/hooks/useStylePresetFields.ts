import { useCallback, useMemo } from 'react';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useAppSelector } from '../../../app/store/storeHooks';
import { useDebouncedMetadata } from '../../../services/api/hooks/useDebouncedMetadata';
import { handlers } from '../../metadata/util/handlers';
import { useImageUrlToBlob } from '../../../common/hooks/useImageUrlToBlob';


export const useStylePresetFields = (preset: StylePresetRecordWithImage | null) => {
    const createPresetFromImage = useAppSelector(s => s.stylePresetModal.createPresetFromImage)

    const imageUrlToBlob = useImageUrlToBlob();

    const getStylePresetFieldDefaults = useCallback(async () => {
        if (preset) {
            let file: File | null = null;
            if (preset.image) {
                const blob = await imageUrlToBlob(preset.image);
                if (blob) {
                    file = new File([blob], "name");
                }

            }

            return {
                name: preset.name,
                positivePrompt: preset.preset_data.positive_prompt || "",
                negativePrompt: preset.preset_data.negative_prompt || "",
                image: file
            };
        }


        return {
            name: "",
            positivePrompt: "",
            negativePrompt: "",
            image: null
        };
    }, [
        preset
    ]);

    return getStylePresetFieldDefaults;
};
