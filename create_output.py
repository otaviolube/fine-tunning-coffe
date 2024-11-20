
from dotenv import load_dotenv
import os
import base64
import requests
import json

# Carregar variáveis do arquivo .env
load_dotenv()

# OpenAI API Key
api_key = os.getenv("API_KEY")
api_key_grok = os.getenv("grok_api_key")

if not api_key or not api_key_grok:
    raise ValueError("API_KEY não encontrada no arquivo .env")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# Path to your images folder
folder_path = "./dataset/images/"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

headers_grok = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key_grok}"
}

# Question to ask for each image
question = """
Você agora é um especialista em agronomia, especializado na análise de doenças em folhas de café. Sua tarefa é identificar se a folha está saudável ou não, qual doença ela apresenta (se houver) e descrever as características observadas. Eu enviarei uma descrição ou imagem (caso aplique) de uma folha de café, e você deverá retornar as informações em um formato JSON.

Por favor, siga este modelo de resposta:

{
  "healthy": true,
  "disease": null,
  "characteristics": null
}

Caso a folha apresente sinais de doença, o JSON deve seguir este formato:

{
  "healthy": false,
  "disease": "Nome da Doença",
  "characteristics": "Descrição detalhada das características observadas na folha."
}

### Exemplos de doenças que você pode identificar:
- Ferrugem do café (Hemileia vastatrix): manchas amareladas ou alaranjadas na parte inferior da folha.
- Cercosporiose (Cercospora coffeicola): manchas circulares com centros claros e bordas escuras.
- Mancha de ascoquita (Ascochyta spp.): manchas irregulares marrons ou pretas.
- Antracnose (Colletotrichum spp.): manchas pretas ou marrons com margens bem definidas, frequentemente com áreas mortas.

**Instruções adicionais:**
- Baseie-se na descrição ou imagem fornecida.
- Se houver múltiplas doenças possíveis, liste a mais provável com base nas características descritas.
- Caso não seja possível determinar com clareza, indique como "inconclusivo" no campo `disease`.

Abaixo segue um exemplo de entrada e saída para o contexto:

Entrada:
"Uma folha de café com manchas amarelas e alaranjadas na parte inferior."

Saída:
{
  "healthy": false,
  "disease": "Ferrugem do café (Hemileia vastatrix)",
  "characteristics": "Manchas amareladas e alaranjadas localizadas na parte inferior da folha."
}

Agora, sempre retorne o diagnóstico em JSON conforme descrito. Aqui está minha análise:

Descrição da folha:
[Descreva a folha ou insira a imagem]
""" #change this depending on your use case

# Function to process each image and get the description
def process_image(image_path, image_name):
    base64_image = encode_image(image_path)
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json = response.json()
    return response_json['choices'][0]['message']['content']

# Function to process each image and get the description - Grok
def process_image_grok(image_path, image_name):
    base64_image = encode_image(image_path)
    
    payload = {
        "model": "grok-vision-beta",
        "stream": False,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers_grok, json=payload)
    response_json = response.json()
    print(response_json)
    return response_json['choices'][0]['message']['content']



# List to store all JSON data
all_json_data = []

# Process each image in the folder
for image_name in os.listdir(folder_path):
    if image_name.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, image_name)
        formatted_answers = process_image_grok(image_path, image_name)
        
        json_data = {
            "id": image_name.split('.')[0],
            "image": image_name,
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": formatted_answers
                }
            ]
        }
        
        all_json_data.append(json_data)
        
# Save the results to a JSON file
output_file = "output_grok.json"
with open(output_file, "w") as outfile:
    json.dump(all_json_data, outfile, indent=4)

print(f"Data has been saved to {output_file}")