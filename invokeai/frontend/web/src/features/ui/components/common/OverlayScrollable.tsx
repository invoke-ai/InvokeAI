import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { PropsWithChildren, memo } from 'react';

const OverlayScrollable = (props: PropsWithChildren) => {
  return (
    <OverlayScrollbarsComponent
      defer
      style={{ height: '100%', width: '100%' }}
      options={{
        scrollbars: {
          visibility: 'visible',
          autoHide: 'never',
          autoHideDelay: 500,
          theme: 'os-theme-dark',
        },
        overflow: { x: 'hidden' },
      }}
    >
      {props.children}
    </OverlayScrollbarsComponent>
  );
};

export default memo(OverlayScrollable);
