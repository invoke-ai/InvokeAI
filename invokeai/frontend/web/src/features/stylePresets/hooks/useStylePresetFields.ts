import { useMemo } from 'react';
import type { StylePresetRecordDTO } from 'services/api/endpoints/stylePresets';


export const useStylePresetFields = (preset: StylePresetRecordDTO | null) => {
    const stylePresetFieldDefaults = useMemo(() => {
        return {
            name: preset ? preset.name : '',
            positivePrompt: preset ? preset.preset_data.positive_prompt : '',
            negativePrompt: preset ? preset.preset_data.negative_prompt : ''
        };
    }, [
        preset
    ]);

    return stylePresetFieldDefaults;
};
