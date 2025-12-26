# GitHub上传指南

## 当前状态
- ✅ Git仓库已初始化
- ✅ .gitignore已配置
- ⚠️ 需要安装Git或使用VS Code Git功能

## 方法一：使用VS Code内置Git（推荐）✨

### 步骤：
1. **打开源代码管理**
   - 快捷键：`Ctrl + Shift + G`
   - 或点击左侧边栏的分支图标

2. **暂存更改**
   - 点击"更改"旁的 `+` 号
   - 或右键选择"暂存所有更改"

3. **提交更改**
   - 在消息框输入：`更新项目：添加ONNX模型转换和Web部署支持`
   - 点击 ✓（提交）按钮

4. **推送到GitHub**
   - 如果是首次推送，VS Code会提示配置远程仓库
   - 输入远程URL：`https://github.com/zlxxxw/GSE-Detection-Portable.git`
   - 点击"推送"按钮

---

## 方法二：安装Git命令行

### Windows安装Git：

**选项A：使用winget（推荐）**
```powershell
winget install --id Git.Git -e --source winget
```

**选项B：手动下载**
1. 访问：https://git-scm.com/download/win
2. 下载并安装
3. **重启VS Code终端**

### 安装后执行：

```powershell
# 进入项目目录
cd "d:\Allen\SoftWare\VS Code\Code\Python\GSE_Detection_Portable"

# 配置Git用户信息（首次使用）
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的GitHub邮箱"

# 检查当前状态
git status

# 添加所有文件
git add .

# 提交更改
git commit -m "更新项目：添加ONNX模型转换和Web部署支持"

# 查看远程仓库（如果已配置）
git remote -v

# 如果没有远程仓库，添加它
git remote add origin https://github.com/zlxxxw/GSE-Detection-Portable.git

# 或者如果远程仓库已存在但URL不对，更新它
git remote set-url origin https://github.com/zlxxxw/GSE-Detection-Portable.git

# 推送到GitHub（首次推送）
git push -u origin main

# 如果分支是master而不是main
git push -u origin master

# 或者强制推送（如果需要覆盖远程仓库）
git push -f origin main
```

---

## 方法三：使用GitHub Desktop

1. 下载：https://desktop.github.com/
2. 安装并登录GitHub账号
3. 选择"Add Local Repository"
4. 选择项目文件夹：`d:\Allen\SoftWare\VS Code\Code\Python\GSE_Detection_Portable`
5. 提交更改并推送

---

## 📋 推送前检查清单

- ✅ 所有重要文件已添加
- ✅ .gitignore已正确配置（避免上传缓存和大文件）
- ✅ README.md包含完整说明
- ✅ requirements.txt已更新
- ✅ 模型文件存在：
  - `weights/gse_detection_v11.pt` (约36MB)
  - `onnx_model/model.onnx` (36.18MB)

---

## ⚠️ 注意事项

### 大文件处理
如果模型文件太大（>100MB），GitHub可能会拒绝推送。解决方案：

1. **使用Git LFS（推荐）**
```powershell
# 安装Git LFS
git lfs install

# 跟踪大文件
git lfs track "*.pt"
git lfs track "*.onnx"

# 添加.gitattributes
git add .gitattributes

# 提交并推送
git add .
git commit -m "添加Git LFS支持"
git push
```

2. **或者在.gitignore中排除模型文件**
```
# 如果模型太大，可以添加到.gitignore
# weights/*.pt
# onnx_model/*.onnx
```

然后在README.md中说明如何下载模型。

---

## 🔍 常见问题

### Q: Git命令不可用？
A: 确保已安装Git并重启终端/VS Code

### Q: 推送被拒绝？
A: 可能需要先拉取远程更改：
```powershell
git pull origin main --rebase
git push origin main
```

### Q: 认证失败？
A: GitHub已不支持密码认证，需要使用：
- Personal Access Token
- SSH密钥
- GitHub Desktop自动处理认证

---

## 📞 需要帮助？

如果遇到问题，请提供：
1. 执行的命令
2. 错误信息
3. 当前Git状态（`git status`输出）
