import type { SystemStyleObject } from '@chakra-ui/react';
import { Badge, CircularProgress, Flex, Icon, Image } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import type { NodeExecutionState } from 'features/nodes/types/invocation';
import { zNodeStatus } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCheck, FaEllipsisH, FaExclamation } from 'react-icons/fa';

type Props = {
  nodeId: string;
};

const iconBoxSize = 3;
const circleStyles: SystemStyleObject = {
  circle: {
    transitionProperty: 'none',
    transitionDuration: '0s',
  },
  '.chakra-progress__track': { stroke: 'transparent' },
};

const InvocationNodeStatusIndicator = ({ nodeId }: Props) => {
  const selectNodeExecutionState = useMemo(
    () =>
      createMemoizedSelector(
        stateSelector,
        ({ nodes }) => nodes.nodeExecutionStates[nodeId]
      ),
    [nodeId]
  );

  const nodeExecutionState = useAppSelector(selectNodeExecutionState);

  if (!nodeExecutionState) {
    return null;
  }

  return (
    <InvTooltip
      label={<TooltipLabel nodeExecutionState={nodeExecutionState} />}
      placement="top"
    >
      <Flex
        className={DRAG_HANDLE_CLASSNAME}
        w={5}
        h="full"
        alignItems="center"
        justifyContent="flex-end"
      >
        <StatusIcon nodeExecutionState={nodeExecutionState} />
      </Flex>
    </InvTooltip>
  );
};

export default memo(InvocationNodeStatusIndicator);

type TooltipLabelProps = {
  nodeExecutionState: NodeExecutionState;
};

const TooltipLabel = memo(({ nodeExecutionState }: TooltipLabelProps) => {
  const { status, progress, progressImage } = nodeExecutionState;
  const { t } = useTranslation();
  if (status === zNodeStatus.enum.PENDING) {
    return <InvText>{t('queue.pending')}</InvText>;
  }
  if (status === zNodeStatus.enum.IN_PROGRESS) {
    if (progressImage) {
      return (
        <Flex pos="relative" pt={1.5} pb={0.5}>
          <Image
            src={progressImage.dataURL}
            w={32}
            h={32}
            borderRadius="base"
            objectFit="contain"
          />
          {progress !== null && (
            <Badge variant="solid" pos="absolute" top={2.5} insetInlineEnd={1}>
              {Math.round(progress * 100)}%
            </Badge>
          )}
        </Flex>
      );
    }

    if (progress !== null) {
      return (
        <InvText>
          {t('nodes.executionStateInProgress')} ({Math.round(progress * 100)}%)
        </InvText>
      );
    }

    return <InvText>{t('nodes.executionStateInProgress')}</InvText>;
  }

  if (status === zNodeStatus.enum.COMPLETED) {
    return <InvText>{t('nodes.executionStateCompleted')}</InvText>;
  }

  if (status === zNodeStatus.enum.FAILED) {
    return <InvText>{t('nodes.executionStateError')}</InvText>;
  }

  return null;
});

TooltipLabel.displayName = 'TooltipLabel';

type StatusIconProps = {
  nodeExecutionState: NodeExecutionState;
};

const StatusIcon = memo((props: StatusIconProps) => {
  const { progress, status } = props.nodeExecutionState;
  if (status === zNodeStatus.enum.PENDING) {
    return <Icon as={FaEllipsisH} boxSize={iconBoxSize} color="base.300" />;
  }
  if (status === zNodeStatus.enum.IN_PROGRESS) {
    return progress === null ? (
      <CircularProgress
        isIndeterminate
        size="14px"
        color="base.500"
        thickness={14}
        sx={circleStyles}
      />
    ) : (
      <CircularProgress
        value={Math.round(progress * 100)}
        size="14px"
        color="base.500"
        thickness={14}
        sx={circleStyles}
      />
    );
  }
  if (status === zNodeStatus.enum.COMPLETED) {
    return <Icon as={FaCheck} boxSize={iconBoxSize} color="ok.300" />;
  }
  if (status === zNodeStatus.enum.FAILED) {
    return <Icon as={FaExclamation} boxSize={iconBoxSize} color="error.300" />;
  }
  return null;
});

StatusIcon.displayName = 'StatusIcon';
