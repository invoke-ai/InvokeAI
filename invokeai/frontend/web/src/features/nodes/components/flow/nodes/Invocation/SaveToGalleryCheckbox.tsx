import { Checkbox, Flex, FormControl, FormLabel } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useHasImageOutput } from 'features/nodes/hooks/useHasImageOutput';
import { useIsIntermediate } from 'features/nodes/hooks/useIsIntermediate';
import { nodeIsIntermediateChanged } from 'features/nodes/store/nodesSlice';
import { ChangeEvent, memo, useCallback } from 'react';

const SaveToGalleryCheckbox = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const hasImageOutput = useHasImageOutput(nodeId);
  const isIntermediate = useIsIntermediate(nodeId);
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        nodeIsIntermediateChanged({
          nodeId,
          isIntermediate: !e.target.checked,
        })
      );
    },
    [dispatch, nodeId]
  );

  if (!hasImageOutput) {
    return null;
  }

  return (
    <FormControl as={Flex} sx={{ alignItems: 'center', gap: 2, w: 'auto' }}>
      <FormLabel sx={{ fontSize: 'xs', mb: '1px' }}>Save to Gallery</FormLabel>
      <Checkbox
        className="nopan"
        size="sm"
        onChange={handleChange}
        isChecked={!isIntermediate}
      />
    </FormControl>
  );
};

export default memo(SaveToGalleryCheckbox);
