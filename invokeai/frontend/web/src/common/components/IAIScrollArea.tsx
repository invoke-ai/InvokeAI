import { ScrollArea, ScrollAreaProps } from '@mantine/core';

type IAIScrollArea = ScrollAreaProps;

export default function IAIScrollArea(props: IAIScrollArea) {
  const { offsetScrollbars = true, ...rest } = props;
  return (
    <ScrollArea offsetScrollbars={offsetScrollbars} {...rest}>
      {props.children}
    </ScrollArea>
  );
}
