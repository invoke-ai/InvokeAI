import {
  Badge,
  CircularProgress,
  Flex,
  Icon,
  Image,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { DRAG_HANDLE_CLASSNAME } from 'features/nodes/types/constants';
import {
  NodeExecutionState,
  zNodeStatus,
} from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCheck, FaEllipsisH, FaExclamation } from 'react-icons/fa';

type Props = {
  nodeId: string;
};

const iconBoxSize = 3;
const circleStyles = {
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
    <Tooltip
      label={<TooltipLabel nodeExecutionState={nodeExecutionState} />}
      placement="top"
    >
      <Flex
        className={DRAG_HANDLE_CLASSNAME}
        sx={{
          w: 5,
          h: 'full',
          alignItems: 'center',
          justifyContent: 'flex-end',
        }}
      >
        <StatusIcon nodeExecutionState={nodeExecutionState} />
      </Flex>
    </Tooltip>
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
    return <Text>{t('queue.pending')}</Text>;
  }
  if (status === zNodeStatus.enum.IN_PROGRESS) {
    if (progressImage) {
      return (
        <Flex sx={{ pos: 'relative', pt: 1.5, pb: 0.5 }}>
          <Image
            src={progressImage.dataURL}
            sx={{ w: 32, h: 32, borderRadius: 'base', objectFit: 'contain' }}
          />
          {progress !== null && (
            <Badge
              variant="solid"
              sx={{ pos: 'absolute', top: 2.5, insetInlineEnd: 1 }}
            >
              {Math.round(progress * 100)}%
            </Badge>
          )}
        </Flex>
      );
    }

    if (progress !== null) {
      return (
        <Text>
          {t('nodes.executionStateInProgress')} ({Math.round(progress * 100)}%)
        </Text>
      );
    }

    return <Text>{t('nodes.executionStateInProgress')}</Text>;
  }

  if (status === zNodeStatus.enum.COMPLETED) {
    return <Text>{t('nodes.executionStateCompleted')}</Text>;
  }

  if (status === zNodeStatus.enum.FAILED) {
    return <Text>{t('nodes.executionStateError')}</Text>;
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
    return (
      <Icon
        as={FaEllipsisH}
        sx={{
          boxSize: iconBoxSize,
          color: 'base.600',
          _dark: { color: 'base.300' },
        }}
      />
    );
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
    return (
      <Icon
        as={FaCheck}
        sx={{
          boxSize: iconBoxSize,
          color: 'ok.600',
          _dark: { color: 'ok.300' },
        }}
      />
    );
  }
  if (status === zNodeStatus.enum.FAILED) {
    return (
      <Icon
        as={FaExclamation}
        sx={{
          boxSize: iconBoxSize,
          color: 'error.600',
          _dark: { color: 'error.300' },
        }}
      />
    );
  }
  return null;
});

StatusIcon.displayName = 'StatusIcon';
