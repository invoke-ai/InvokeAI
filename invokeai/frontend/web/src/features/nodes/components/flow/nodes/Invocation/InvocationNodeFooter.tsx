import {
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  Spacer,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useHasImageOutput } from 'features/nodes/hooks/useHasImageOutput';
import { useIsIntermediate } from 'features/nodes/hooks/useIsIntermediate';
import { fieldBooleanValueChanged } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { ChangeEvent, memo, useCallback } from 'react';

type Props = {
  nodeId: string;
};

const InvocationNodeFooter = ({ nodeId }: Props) => {
  return (
    <Flex
      className={DRAG_HANDLE_CLASSNAME}
      layerStyle="nodeFooter"
      sx={{
        w: 'full',
        borderBottomRadius: 'base',
        px: 2,
        py: 0,
        h: 6,
      }}
    >
      <Spacer />
      <SaveImageCheckbox nodeId={nodeId} />
    </Flex>
  );
};

export default memo(InvocationNodeFooter);

const SaveImageCheckbox = memo(({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const hasImageOutput = useHasImageOutput(nodeId);
  const is_intermediate = useIsIntermediate(nodeId);
  const handleChangeIsIntermediate = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        fieldBooleanValueChanged({
          nodeId,
          fieldName: 'is_intermediate',
          value: !e.target.checked,
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
      <FormLabel sx={{ fontSize: 'xs', mb: '1px' }}>Save Output</FormLabel>
      <Checkbox
        className="nopan"
        size="sm"
        onChange={handleChangeIsIntermediate}
        isChecked={!is_intermediate}
      />
    </FormControl>
  );
});

SaveImageCheckbox.displayName = 'SaveImageCheckbox';
