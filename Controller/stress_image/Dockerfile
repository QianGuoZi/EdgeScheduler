# 使用轻量级的Alpine Linux作为基础镜像
FROM alpine:3.18

# 安装stress-ng工具
RUN apk add --no-cache stress-ng

# 设置默认命令（可以被docker-compose覆盖）
CMD ["stress-ng", "--cpu", "1", "--vm", "1", "--vm-bytes", "1G", "--timeout", "0"]
