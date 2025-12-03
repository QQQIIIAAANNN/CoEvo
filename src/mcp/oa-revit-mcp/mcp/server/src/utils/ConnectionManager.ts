import { RevitClientConnection } from "./SocketClient.js";

/**
 * 连接到Revit客户端并执行操作
 * @param operation 连接成功后要执行的操作函数
 * @returns 操作的结果
 */
export async function withRevitConnection<T>(
  operation: (client: RevitClientConnection) => Promise<T>
): Promise<T> {
  const revitClient = new RevitClientConnection("localhost", 8080);

  try {
    // 连接到Revit客户端
    if (!revitClient.isConnected) {
      await new Promise<void>((resolve, reject) => {
        const onConnect = () => {
          revitClient.socket.removeListener("connect", onConnect);
          revitClient.socket.removeListener("error", onError);
          resolve();
        };

        const onError = (error: any) => {
          revitClient.socket.removeListener("connect", onConnect);
          revitClient.socket.removeListener("error", onError);
          reject(new Error("connect to revit client failed"));
        };

        revitClient.socket.on("connect", onConnect);
        revitClient.socket.on("error", onError);

        revitClient.connect();

        setTimeout(() => {
          revitClient.socket.removeListener("connect", onConnect);
          revitClient.socket.removeListener("error", onError);
          reject(new Error("连接到Revit客户端失败"));
        }, 5000);
      });
    }

    // 执行操作
    return await operation(revitClient);
  } finally {
    // 断开连接
    revitClient.disconnect();
  }
}
