import {
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  Spacer,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldBooleanValueChanged } from 'features/nodes/store/nodesSlice';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import {
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { some } from 'lodash-es';
import { ChangeEvent, memo, useCallback, useMemo } from 'react';
import { NodeProps } from 'reactflow';

export const IMAGE_FIELDS = ['ImageField', 'ImageCollection'];
export const FOOTER_FIELDS = IMAGE_FIELDS;

type Props = {
  nodeProps: NodeProps<InvocationNodeData>;
  nodeTemplate: InvocationTemplate;
};

const NodeFooter = (props: Props) => {
  const { nodeProps, nodeTemplate } = props;
  const dispatch = useAppDispatch();

  const hasImageOutput = useMemo(
    () =>
      some(nodeTemplate?.outputs, (output) =>
        IMAGE_FIELDS.includes(output.type)
      ),
    [nodeTemplate?.outputs]
  );

  const handleChangeIsIntermediate = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        fieldBooleanValueChanged({
          nodeId: nodeProps.data.id,
          fieldName: 'is_intermediate',
          value: !e.target.checked,
        })
      );
    },
    [dispatch, nodeProps.data.id]
  );

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
      {hasImageOutput && (
        <FormControl as={Flex} sx={{ alignItems: 'center', gap: 2, w: 'auto' }}>
          <FormLabel sx={{ fontSize: 'xs', mb: '1px' }}>Save Output</FormLabel>
          <Checkbox
            className="nopan"
            size="sm"
            onChange={handleChangeIsIntermediate}
            isChecked={!nodeProps.data.inputs['is_intermediate']?.value}
          />
        </FormControl>
      )}
    </Flex>
  );
};

export default memo(NodeFooter);
