import 'reactflow/dist/style.css';
import { useCallback, forwardRef } from 'react';
import { Flex, Text } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { nodeAdded, nodesSelector } from '../store/nodesSlice';
import { map } from 'lodash-es';
import { useBuildInvocation } from '../hooks/useBuildInvocation';
import { AnyInvocationType } from 'services/events/types';
import { useAppToaster } from 'app/components/Toaster';
import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIMantineMultiSelect from 'common/components/IAIMantineMultiSelect';

type NodeTemplate = {
  label: string;
  value: string;
  description: string;
};

const selector = createSelector(
  nodesSelector,
  (nodes) => {
    const data: NodeTemplate[] = map(nodes.invocationTemplates, (template) => {
      return {
        label: template.title,
        value: template.type,
        description: template.description,
      };
    });

    return { data };
  },
  defaultSelectorOptions
);

const AddNodeMenu = () => {
  const dispatch = useAppDispatch();
  const { data } = useAppSelector(selector);

  const buildInvocation = useBuildInvocation();

  const toaster = useAppToaster();

  const addNode = useCallback(
    (nodeType: AnyInvocationType) => {
      const invocation = buildInvocation(nodeType);

      if (!invocation) {
        toaster({
          status: 'error',
          title: `Unknown Invocation type ${nodeType}`,
        });
        return;
      }

      dispatch(nodeAdded(invocation));
    },
    [dispatch, buildInvocation, toaster]
  );

  return (
    <Flex sx={{ gap: 2, alignItems: 'center' }}>
      <IAIMantineMultiSelect
        selectOnBlur={false}
        placeholder="Add Node"
        value={[]}
        data={data}
        maxDropdownHeight={400}
        nothingFound="No matching nodes"
        itemComponent={SelectItem}
        filter={(value, selected, item: NodeTemplate) =>
          item.label.toLowerCase().includes(value.toLowerCase().trim()) ||
          item.value.toLowerCase().includes(value.toLowerCase().trim()) ||
          item.description.toLowerCase().includes(value.toLowerCase().trim())
        }
        onChange={(v) => {
          v[0] && addNode(v[0] as AnyInvocationType);
        }}
        sx={{
          width: '18rem',
        }}
      />
    </Flex>
  );
};

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  value: string;
  label: string;
  description: string;
}

const SelectItem = forwardRef<HTMLDivElement, ItemProps>(
  ({ label, description, ...others }: ItemProps, ref) => {
    return (
      <div ref={ref} {...others}>
        <div>
          <Text>{label}</Text>
          <Text size="xs" color="base.600">
            {description}
          </Text>
        </div>
      </div>
    );
  }
);

SelectItem.displayName = 'SelectItem';

export default AddNodeMenu;
