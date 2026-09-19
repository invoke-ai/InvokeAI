import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import { memo } from 'react';

import { NodeSettingFooterControl } from './NodeSettingFooterControl';

type Props = {
  nodeId: string;
};

const sx: SystemStyleObject = {
  w: 'full',
  borderBottomRadius: 'base',
  // One row per setting, so each connection handle lines up with the label it belongs to
  flexDir: 'column',
  px: 2,
  py: 1,
  // The add/remove form element buttons are hidden by default and shown on hover
  '& .node-setting-action-button': {
    display: 'none',
  },
  _hover: {
    '& .node-setting-action-button': {
      display: 'inline-flex',
    },
  },
};

const InvocationNodeFooter = ({ nodeId }: Props) => {
  return (
    <Flex className={DRAG_HANDLE_CLASSNAME} layerStyle="nodeFooter" sx={sx}>
      <NodeSettingFooterControl nodeId={nodeId} setting="use_cache" />
      <NodeSettingFooterControl nodeId={nodeId} setting="save_to_gallery" />
    </Flex>
  );
};

export default memo(InvocationNodeFooter);
