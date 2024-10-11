import { EMPTY_ARRAY } from "app/store/constants";
import { useCallback,useMemo } from "react";
import { modelConfigsAdapterSelectors,useGetModelConfigsQuery } from "services/api/endpoints/models";
import type { StarterModel } from "services/api/types";

export const useBuildModelsToInstall = () => {
    const { data: modelListRes } = useGetModelConfigsQuery();
    const modelList = useMemo(() => {
        if (!modelListRes) {
            return EMPTY_ARRAY;
        }

        return modelConfigsAdapterSelectors.selectAll(modelListRes);
    }, [modelListRes]);

    const buildModelToInstall = useCallback(
        (starterModel: StarterModel) => {
            if (
                modelList.some(
                    (mc) => starterModel.base === mc.base && starterModel.name === mc.name && starterModel.type === mc.type
                )
            ) {
                return undefined;
            }

            const source = starterModel.source;
            const config = {
                name: starterModel.name,
                description: starterModel.description,
                type: starterModel.type,
                base: starterModel.base,
                format: starterModel.format,
            };
            return { config, source };
        },
        [modelList]
    );

    return buildModelToInstall
}