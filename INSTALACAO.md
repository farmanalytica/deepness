# Guia de Instalação do Plugin Deepness

## Pré-requisitos (nesta ordem)

### 1. QGIS 3.44.3 ou posterior
- Baixe em https://qgis.org/download/
- **Caminho personalizado (opcional):**
  - Escolha : `C:\QGIS\`
  - NÂO em: `C:\Program Files\
- Conclua a instalação
- Verifique: Abra QGIS e confirme que carrega

### 2. CUDA + cuDNN 

**CUDA 12.4 ou 13.x:**
- Baixe em https://developer.nvidia.com/cuda-toolkit
- Instale no local padrão
- Verifique: Confira se `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin` existe

**cuDNN 9.x:**
- Baixe em https://developer.nvidia.com/cudnn (requer conta)
- Extraia para `C:\Program Files\NVIDIA\CUDNN\`
- Verifique DLLs em `C:\Program Files\NVIDIA\CUDNN\v9.17\bin\`


---

## Instalar Plugin Deepness

### Passo 1: Abrir Gerenciador de Plugins do QGIS
- QGIS → Extensões → Gerenciar e Instalar Extensões

### Passo 2: Instalar a partir de ZIP
- Clique na aba **"Instalar a partir de ZIP"**
- Selecione o arquivo `deepness.zip`
- Clique em **"Instalar extensão"**
- Aguarde a instalação

### Passo 3: Reinicie QGIS
- Feche QGIS completamente
- Reabra QGIS

### Passo 4: Instalar Pacotes Python
- Quando QGIS inicia, um diálogo "Instalador de Pacotes" aparece
- Clique no botão **"Instalar pacotes"**
- Aguarde a instalação (leva 2-3 minutos)
- Clique em **"Testar e Fechar"**

---

## Verificar Instalação

Abra o Console Python do QGIS e execute:
```python
import cv2
import onnxruntime
print("✓ Dependências do Deepness instaladas com sucesso")
```

---

## Suporte GPU (Opcional)

Após instalação de CUDA/cuDNN, GPU será detectada automaticamente no próximo reinício do QGIS.

Verifique no console do QGIS por:
```
[Deepness GPU Diagnostics] onnxruntime available providers: [...'CUDAExecutionProvider',...]
[SUCCESS] Using CUDAExecutionProvider
```

Se ainda usar CPU: reinstale `onnxruntime-gpu`
```python
!pip uninstall onnxruntime-gpu
!pip install onnxruntime-gpu
```

---

## Solução de Problemas

### Plugin não carrega
- Verifique versão QGIS ≥ 3.44.3
- Confirme estrutura de pasta: `plugins/deepness/__init__.py` existe
- Verifique se caminhos não contêm caracteres especiais

### CUDA não é detectado
- Reinicie QGIS após instalar CUDA
- Verifique caminho CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
- Confirme cuDNN extraído corretamente

### Instalação de pacotes falha
- Tente novamente no console Python do QGIS:
  ```python
  !pip install --upgrade onnxruntime-gpu opencv-python-headless
  ```

---

## Suporte
Reporte problemas em: https://github.com/farmanalytica/deepness/issues
