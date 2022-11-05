// vite.config.ts
import { defineConfig } from "file:///D:/Desktop%20D/Temp%20Stuff/Programming%20Projects/Deep%20Learning/StableDiffusion/bless/InvokeAI/frontend/node_modules/vite/dist/node/index.js";
import react from "file:///D:/Desktop%20D/Temp%20Stuff/Programming%20Projects/Deep%20Learning/StableDiffusion/bless/InvokeAI/frontend/node_modules/@vitejs/plugin-react/dist/index.mjs";
import eslint from "file:///D:/Desktop%20D/Temp%20Stuff/Programming%20Projects/Deep%20Learning/StableDiffusion/bless/InvokeAI/frontend/node_modules/vite-plugin-eslint/dist/index.mjs";
import tsconfigPaths from "file:///D:/Desktop%20D/Temp%20Stuff/Programming%20Projects/Deep%20Learning/StableDiffusion/bless/InvokeAI/frontend/node_modules/vite-tsconfig-paths/dist/index.mjs";
var vite_config_default = defineConfig(({ mode }) => {
  const common = {
    base: "",
    plugins: [react(), eslint(), tsconfigPaths()],
    server: {
      proxy: {
        "/outputs": {
          target: "http://127.0.0.1:9090/outputs",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/outputs/, "")
        },
        "/flaskwebgui-keep-server-alive": {
          target: "http://127.0.0.1:9090/flaskwebgui-keep-server-alive",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/flaskwebgui-keep-server-alive/, "")
        },
        "/socket.io": {
          target: "ws://127.0.0.1:9090",
          ws: true
        }
      }
    },
    build: {
      target: "esnext",
      chunkSizeWarningLimit: 1500
    }
  };
  if (mode == "development") {
    return {
      ...common,
      build: {
        ...common.build
      }
    };
  } else {
    return {
      ...common
    };
  }
});
export {
  vite_config_default as default
};
//# sourceMappingURL=data:application/json;base64,ewogICJ2ZXJzaW9uIjogMywKICAic291cmNlcyI6IFsidml0ZS5jb25maWcudHMiXSwKICAic291cmNlc0NvbnRlbnQiOiBbImNvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9kaXJuYW1lID0gXCJEOlxcXFxEZXNrdG9wIERcXFxcVGVtcCBTdHVmZlxcXFxQcm9ncmFtbWluZyBQcm9qZWN0c1xcXFxEZWVwIExlYXJuaW5nXFxcXFN0YWJsZURpZmZ1c2lvblxcXFxibGVzc1xcXFxJbnZva2VBSVxcXFxmcm9udGVuZFwiO2NvbnN0IF9fdml0ZV9pbmplY3RlZF9vcmlnaW5hbF9maWxlbmFtZSA9IFwiRDpcXFxcRGVza3RvcCBEXFxcXFRlbXAgU3R1ZmZcXFxcUHJvZ3JhbW1pbmcgUHJvamVjdHNcXFxcRGVlcCBMZWFybmluZ1xcXFxTdGFibGVEaWZmdXNpb25cXFxcYmxlc3NcXFxcSW52b2tlQUlcXFxcZnJvbnRlbmRcXFxcdml0ZS5jb25maWcudHNcIjtjb25zdCBfX3ZpdGVfaW5qZWN0ZWRfb3JpZ2luYWxfaW1wb3J0X21ldGFfdXJsID0gXCJmaWxlOi8vL0Q6L0Rlc2t0b3AlMjBEL1RlbXAlMjBTdHVmZi9Qcm9ncmFtbWluZyUyMFByb2plY3RzL0RlZXAlMjBMZWFybmluZy9TdGFibGVEaWZmdXNpb24vYmxlc3MvSW52b2tlQUkvZnJvbnRlbmQvdml0ZS5jb25maWcudHNcIjtpbXBvcnQgeyBkZWZpbmVDb25maWcgfSBmcm9tICd2aXRlJztcclxuaW1wb3J0IHJlYWN0IGZyb20gJ0B2aXRlanMvcGx1Z2luLXJlYWN0JztcclxuaW1wb3J0IGVzbGludCBmcm9tICd2aXRlLXBsdWdpbi1lc2xpbnQnO1xyXG5pbXBvcnQgdHNjb25maWdQYXRocyBmcm9tICd2aXRlLXRzY29uZmlnLXBhdGhzJztcclxuXHJcbi8vIGh0dHBzOi8vdml0ZWpzLmRldi9jb25maWcvXHJcbmV4cG9ydCBkZWZhdWx0IGRlZmluZUNvbmZpZygoeyBtb2RlIH0pID0+IHtcclxuICBjb25zdCBjb21tb24gPSB7XHJcbiAgICBiYXNlOiAnJyxcclxuICAgIHBsdWdpbnM6IFtyZWFjdCgpLCBlc2xpbnQoKSwgdHNjb25maWdQYXRocygpXSxcclxuICAgIHNlcnZlcjoge1xyXG4gICAgICAvLyBQcm94eSBIVFRQIHJlcXVlc3RzIHRvIHRoZSBmbGFzayBzZXJ2ZXJcclxuICAgICAgcHJveHk6IHtcclxuICAgICAgICAnL291dHB1dHMnOiB7XHJcbiAgICAgICAgICB0YXJnZXQ6ICdodHRwOi8vMTI3LjAuMC4xOjkwOTAvb3V0cHV0cycsXHJcbiAgICAgICAgICBjaGFuZ2VPcmlnaW46IHRydWUsXHJcbiAgICAgICAgICByZXdyaXRlOiAocGF0aCkgPT4gcGF0aC5yZXBsYWNlKC9eXFwvb3V0cHV0cy8sICcnKSxcclxuICAgICAgICB9LFxyXG4gICAgICAgICcvZmxhc2t3ZWJndWkta2VlcC1zZXJ2ZXItYWxpdmUnOiB7XHJcbiAgICAgICAgICB0YXJnZXQ6ICdodHRwOi8vMTI3LjAuMC4xOjkwOTAvZmxhc2t3ZWJndWkta2VlcC1zZXJ2ZXItYWxpdmUnLFxyXG4gICAgICAgICAgY2hhbmdlT3JpZ2luOiB0cnVlLFxyXG4gICAgICAgICAgcmV3cml0ZTogKHBhdGgpID0+XHJcbiAgICAgICAgICAgIHBhdGgucmVwbGFjZSgvXlxcL2ZsYXNrd2ViZ3VpLWtlZXAtc2VydmVyLWFsaXZlLywgJycpLFxyXG4gICAgICAgIH0sXHJcbiAgICAgICAgLy8gUHJveHkgc29ja2V0LmlvIHRvIHRoZSBmbGFzay1zb2NrZXRpbyBzZXJ2ZXJcclxuICAgICAgICAnL3NvY2tldC5pbyc6IHtcclxuICAgICAgICAgIHRhcmdldDogJ3dzOi8vMTI3LjAuMC4xOjkwOTAnLFxyXG4gICAgICAgICAgd3M6IHRydWUsXHJcbiAgICAgICAgfSxcclxuICAgICAgfSxcclxuICAgIH0sXHJcbiAgICBidWlsZDoge1xyXG4gICAgICB0YXJnZXQ6ICdlc25leHQnLFxyXG4gICAgICBjaHVua1NpemVXYXJuaW5nTGltaXQ6IDE1MDAsIC8vIHdlIGRvbid0IHJlYWxseSBjYXJlIGFib3V0IGNodW5rIHNpemVcclxuICAgIH0sXHJcbiAgfTtcclxuICBpZiAobW9kZSA9PSAnZGV2ZWxvcG1lbnQnKSB7XHJcbiAgICByZXR1cm4ge1xyXG4gICAgICAuLi5jb21tb24sXHJcbiAgICAgIGJ1aWxkOiB7XHJcbiAgICAgICAgLi4uY29tbW9uLmJ1aWxkLFxyXG4gICAgICAgIC8vIHNvdXJjZW1hcDogdHJ1ZSwgLy8gdGhpcyBjYW4gYmUgZW5hYmxlZCBpZiBuZWVkZWQsIGl0IGFkZHMgb3Z3ZXIgMTVNQiB0byB0aGUgY29tbWl0XHJcbiAgICAgIH0sXHJcbiAgICB9O1xyXG4gIH0gZWxzZSB7XHJcbiAgICByZXR1cm4ge1xyXG4gICAgICAuLi5jb21tb24sXHJcbiAgICB9O1xyXG4gIH1cclxufSk7XHJcbiJdLAogICJtYXBwaW5ncyI6ICI7QUFBa2YsU0FBUyxvQkFBb0I7QUFDL2dCLE9BQU8sV0FBVztBQUNsQixPQUFPLFlBQVk7QUFDbkIsT0FBTyxtQkFBbUI7QUFHMUIsSUFBTyxzQkFBUSxhQUFhLENBQUMsRUFBRSxLQUFLLE1BQU07QUFDeEMsUUFBTSxTQUFTO0FBQUEsSUFDYixNQUFNO0FBQUEsSUFDTixTQUFTLENBQUMsTUFBTSxHQUFHLE9BQU8sR0FBRyxjQUFjLENBQUM7QUFBQSxJQUM1QyxRQUFRO0FBQUEsTUFFTixPQUFPO0FBQUEsUUFDTCxZQUFZO0FBQUEsVUFDVixRQUFRO0FBQUEsVUFDUixjQUFjO0FBQUEsVUFDZCxTQUFTLENBQUMsU0FBUyxLQUFLLFFBQVEsY0FBYyxFQUFFO0FBQUEsUUFDbEQ7QUFBQSxRQUNBLGtDQUFrQztBQUFBLFVBQ2hDLFFBQVE7QUFBQSxVQUNSLGNBQWM7QUFBQSxVQUNkLFNBQVMsQ0FBQyxTQUNSLEtBQUssUUFBUSxvQ0FBb0MsRUFBRTtBQUFBLFFBQ3ZEO0FBQUEsUUFFQSxjQUFjO0FBQUEsVUFDWixRQUFRO0FBQUEsVUFDUixJQUFJO0FBQUEsUUFDTjtBQUFBLE1BQ0Y7QUFBQSxJQUNGO0FBQUEsSUFDQSxPQUFPO0FBQUEsTUFDTCxRQUFRO0FBQUEsTUFDUix1QkFBdUI7QUFBQSxJQUN6QjtBQUFBLEVBQ0Y7QUFDQSxNQUFJLFFBQVEsZUFBZTtBQUN6QixXQUFPO0FBQUEsTUFDTCxHQUFHO0FBQUEsTUFDSCxPQUFPO0FBQUEsUUFDTCxHQUFHLE9BQU87QUFBQSxNQUVaO0FBQUEsSUFDRjtBQUFBLEVBQ0YsT0FBTztBQUNMLFdBQU87QUFBQSxNQUNMLEdBQUc7QUFBQSxJQUNMO0FBQUEsRUFDRjtBQUNGLENBQUM7IiwKICAibmFtZXMiOiBbXQp9Cg==
