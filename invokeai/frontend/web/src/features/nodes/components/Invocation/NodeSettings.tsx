import { Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIPopover from 'common/components/IAIPopover';
import IAISwitch from 'common/components/IAISwitch';
import { fieldBooleanValueChanged } from 'features/nodes/store/nodesSlice';
import { InvocationNodeData } from 'features/nodes/types/types';
import { ChangeEvent, memo, useCallback } from 'react';
import { FaBars } from 'react-icons/fa';

interface Props {
  data: InvocationNodeData;
}

const NodeSettings = (props: Props) => {
  const { data } = props;
  const dispatch = useAppDispatch();

  const handleChangeIsIntermediate = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        fieldBooleanValueChanged({
          nodeId: data.id,
          fieldName: 'is_intermediate',
          value: e.target.checked,
        })
      );
    },
    [data.id, dispatch]
  );

  return (
    <IAIPopover
      isLazy={false}
      triggerComponent={
        <IAIIconButton
          className="nopan"
          aria-label="Node Settings"
          variant="link"
          sx={{
            minW: 8,
            color: 'base.500',
            _dark: {
              color: 'base.500',
            },
            _hover: {
              color: 'base.700',
              _dark: {
                color: 'base.300',
              },
            },
          }}
          icon={<FaBars />}
        />
      }
    >
      <Flex sx={{ flexDir: 'column', gap: 4, w: 64 }}>
        <IAISwitch
          label="Intermediate"
          isChecked={Boolean(data.inputs['is_intermediate']?.value)}
          onChange={handleChangeIsIntermediate}
          helperText="The outputs of intermediate nodes are considered temporary objects. Intermediate images are not added to the gallery."
        />
      </Flex>
    </IAIPopover>
  );
};

export default memo(NodeSettings);
