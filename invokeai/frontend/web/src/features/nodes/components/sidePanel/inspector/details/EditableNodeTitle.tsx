import {
  Editable,
  EditableInput,
  EditablePreview,
  Flex,
  Text,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { useNodeLabel } from 'features/nodes/hooks/useNodeLabel';
import { useNodeTemplateTitle } from 'features/nodes/hooks/useNodeTemplateTitle';
import { useNodeVersion } from 'features/nodes/hooks/useNodeVersion';
import { nodeLabelChanged } from 'features/nodes/store/nodesSlice';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSync } from 'react-icons/fa';

type EditableNodeTitleProps = {
  nodeId?: string;
};

const EditableNodeTitle = (props: EditableNodeTitleProps) => {
  if (!props.nodeId) {
    return (
      <Text
        sx={{
          fontWeight: 600,
          px: 1,
          color: 'base.700',
          _dark: { color: 'base.200' },
        }}
      >
        No node selected
      </Text>
    );
  }

  return (
    <Flex
      sx={{
        justifyContent: 'space-between',
        alignItems: 'center',
        px: 1,
        color: 'base.700',
        _dark: { color: 'base.200' },
      }}
    >
      <EditableTitle nodeId={props.nodeId} />
      <Version nodeId={props.nodeId} />
    </Flex>
  );
};

type VersionProps = {
  nodeId: string;
};

const Version = memo(({ nodeId }: VersionProps) => {
  const { version, needsUpdate, updateNode } = useNodeVersion(nodeId);

  const { t } = useTranslation();

  return (
    <Flex alignItems="center" gap={1}>
      <Text variant={needsUpdate ? 'error' : 'subtext'} fontWeight={600}>
        v{version}
      </Text>
      {needsUpdate && (
        <IAIIconButton
          size="sm"
          aria-label={t('nodes.updateNode')}
          tooltip={t('nodes.updateNode')}
          icon={<FaSync />}
          variant="link"
          onClick={updateNode}
        />
      )}
    </Flex>
  );
});

Version.displayName = 'Version';

type EditableTitleProps = {
  nodeId: string;
};

const EditableTitle = memo(({ nodeId }: EditableTitleProps) => {
  const dispatch = useAppDispatch();
  const label = useNodeLabel(nodeId);
  const templateTitle = useNodeTemplateTitle(nodeId);
  const { t } = useTranslation();

  const [localTitle, setLocalTitle] = useState('');
  const handleSubmit = useCallback(
    async (newTitle: string) => {
      if (!newTitle.trim()) {
        setLocalTitle(label || templateTitle || t('nodes.problemSettingTitle'));
        return;
      }
      dispatch(nodeLabelChanged({ nodeId, label: newTitle }));
      setLocalTitle(label || templateTitle || t('nodes.problemSettingTitle'));
    },
    [dispatch, nodeId, templateTitle, label, t]
  );

  const handleChange = useCallback((newTitle: string) => {
    setLocalTitle(newTitle);
  }, []);

  useEffect(() => {
    // Another component may change the title; sync local title with global state
    setLocalTitle(label || templateTitle || t('nodes.problemSettingTitle'));
  }, [label, templateTitle, t]);

  return (
    <Editable
      as={Flex}
      value={localTitle}
      onChange={handleChange}
      onSubmit={handleSubmit}
      w="full"
    >
      <EditablePreview p={0} fontWeight={600} noOfLines={1} />
      <EditableInput
        p={0}
        className="nodrag"
        fontWeight={700}
        _focusVisible={{ boxShadow: 'none' }}
      />
    </Editable>
  );
});

EditableTitle.displayName = 'EditableTitle';

export default memo(EditableNodeTitle);
