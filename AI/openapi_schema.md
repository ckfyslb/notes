```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "知识库检索",
    "description": "从Ragflow检索知识库",
    "version": "v1.0.0"
  },
  "servers": [
    {
      "url": "http://192.168.2.13/api/v1"
    }
  ],
  "paths": {
    "/retrieval": {
      "post": {
        "summary": "检索文档块",
        "description": "从指定的数据集中检索相关的文档块",
        "operationId": "知识库检索工具",
        "requestBody": {
          "description": "Retrieve request payload",
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "question": {
                    "type": "string",
                    "description": "用于检索信息的问题。"
                  },
                  "dataset_ids": {
                    "type": "array",
                    "items": {
                      "type": "string"
                    },
                    "description": "要检索的数据集 ID 列表。"
                  },
                  "document_ids": {
                    "type": "array",
                    "items": {
                      "type": "string"
                    },
                    "description": "要检索的文档 ID 列表。"
                  }
                },
                "required": ["question"]
              }
            }
          }
        },
        "security": [
          {
            "BearerAuth": []
          }
        ]
      }
    }
  },
  "components": {
    "schemas": {}
  }
}









{
  "openapi":"3.1.0",
  "info":{
    "title":"YuanyanAPI",
    "version":"0.1.0"
  },
  "paths":{
    "/yuanyanAgent/health/":{
      "get":{
        "summary":"健康检查",
        "description":"健康检查接口，用于检查 yuanyan-api 服务是否正常运行。",
        "operationId":"health_yuanyanAgent_health__get",
        "responses":{
          "200":{
            "description":"Successful Response",
            "content":{
              "application/json":{
                "schema":{ }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/llm_newchat/":{
      "post":{
        "summary":"清空历史对话",
        "description":"清空历史对话接口，在新建对话时使用，用户点击新建对话按钮，前端进入新的空聊天界面，后端清空之前的聊天记录，不返回内容。",
        "operationId":"llm_newchat_yuanyanAgent_llm_newchat__post",
        "responses":{
          "200":{
            "description":"Successful Response",
            "content":{
              "application/json":{
                "schema":{ }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/llm/":{
      "post":{
        "summary":"与大模型对话",
        "description":"大模型对话接口：用户输入聊天内容点击发送，同时传输用户勾选的知识库 ID 列表（没有勾选则为空列表），后端流式返回大模型输出。",
        "operationId":"stream_handler_yuanyanAgent_llm__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/ChatRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"流式响应",
            "content":{
              "application/json":{
                "schema":{ }
              },
              "text/event-stream":{ }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/createProject/":{
      "post":{
        "summary":"创建项目",
        "description":"配置项创建一个知识库",
        "operationId":"createProject_yuanyanAgent_createProject__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/createProjectRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"解析成功",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10000":{
            "description":"Ragflow调用失败",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10001":{
            "description":"已存在此名称的数据库",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/deleteProject/":{
      "delete":{
        "summary":"删除项目",
        "description":"删除一个知识库",
        "operationId":"deleteProject_yuanyanAgent_deleteProject__delete",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/deleteProjectRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"删除成功",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10000":{
            "description":"Ragflow调用失败",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10001":{
            "description":"不存在此ID的数据库",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/upload/":{
      "post":{
        "summary":"上传并解析文档和代码",
        "description":"上传并解析文档和代码，返回文件的ID",
        "operationId":"upload_yuanyanAgent_upload__post",
        "requestBody":{
          "content":{
            "multipart/form-data":{
              "schema":{
                "$ref":"#/components/schemas/Body_upload_yuanyanAgent_upload__post"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":" progressing: 进度, 流式响应",
            "content":{
              "application/json":{
                "schema":{ }
              },
              "text/event-stream":{ }
            }
          },
          "10000":{
            "description":"不存在该ID的数据库",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/deleteDocument/":{
      "delete":{
        "summary":"删除文档",
        "description":"删除一个文档",
        "operationId":"deleteDocument_yuanyanAgent_deleteDocument__delete",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/deleteDocumentRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"删除成功",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10000":{
            "description":"Ragflow调用失败",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10001":{
            "description":"不存在此ID的数据库",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/listChunks/":{
      "post":{
        "summary":"列举解析结果",
        "description":"列举解析结果（解析后，文档被分块）",
        "operationId":"listChunks_yuanyanAgent_listChunks__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/listChunksRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"删除成功",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10000":{
            "description":"Ragflow调用失败",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "10001":{
            "description":"不存在此ID的数据库",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/APIResponse"
                }
              }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/extractgongneng/":{
      "post":{
        "summary":"提取功能",
        "description":"将某个需求文档传给Python，Python分析软件的功能后，以流式返回给前端，前端进行字符串分割显示",
        "operationId":"extract_gongneng_yuanyanAgent_extractgongneng__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/extractGNRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"返回一串功能名字，每个名字通过分号隔开。",
            "content":{
              "application/json":{
                "schema":{ }
              },
              "text/event-stream":{ }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/extractxuqiu/":{
      "post":{
        "summary":"提取需求",
        "description":"将某个功能和该功能所在的软件需求规格说明发送给大模型，大模型自动提取需求原文。",
        "operationId":"extract_xuqiu_yuanyanAgent_extractxuqiu__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/extractXuqiuRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"Successful Response",
            "content":{
              "application/json":{
                "schema":{ }
              }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/wendangshencha/":{
      "post":{
        "summary":"文档审查",
        "description":"将extractxuqiu调用后获取的需求原文发给大模型，大模型返回文档审查问题。格式为【问题名称】+【问题描述】",
        "operationId":"wendangshencha_yuanyanAgent_wendangshencha__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/wendangshenchaRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"流式响应",
            "content":{
              "application/json":{
                "schema":{ }
              },
              "text/event-stream":{ }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/xuqiuOptimize/":{
      "post":{
        "summary":"需求优化",
        "description":"将extractxuqiu调用后获取的需求原文发个大模型，大模型根据软件配置项的全部文档，对需求进行优化，返回优化后的需求。",
        "operationId":"xuqiuOptimize_yuanyanAgent_xuqiuOptimize__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/optimizeXuqiuRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"流式响应",
            "content":{
              "application/json":{
                "schema":{ }
              },
              "text/event-stream":{ }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/ceshidagang/":{
      "post":{
        "summary":"生成测试大纲",
        "description":"将优化后的功能描述发送给大模型，大模型分析该功能形成测试大纲。",
        "operationId":"csdg_interface_yuanyanAgent_ceshidagang__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/dagangGenrateRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"流式响应",
            "content":{
              "application/json":{
                "schema":{ }
              },
              "text/event-stream":{ }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    },
    "/yuanyanAgent/ceshiyongli/":{
      "post":{
        "summary":"生成测试用例",
        "description":"将测试大纲发给大模型，生成测试用例。",
        "operationId":"csyl_interface_yuanyanAgent_ceshiyongli__post",
        "requestBody":{
          "content":{
            "application/json":{
              "schema":{
                "$ref":"#/components/schemas/yongliGenrateRequest"
              }
            }
          },
          "required":true
        },
        "responses":{
          "200":{
            "description":"流式响应",
            "content":{
              "application/json":{
                "schema":{ }
              },
              "text/event-stream":{ }
            }
          },
          "422":{
            "description":"Validation Error",
            "content":{
              "application/json":{
                "schema":{
                  "$ref":"#/components/schemas/HTTPValidationError"
                }
              }
            }
          }
        }
      }
    }
  },
  "components":{
    "schemas":{
      "APIResponse":{
        "properties":{
          "code":{
            "type":"integer",
            "title":"Code"
          },
          "status":{
            "type":"string",
            "title":"Status"
          },
          "message":{
            "type":"string",
            "title":"Message"
          },
          "data":{
            "type":"object",
            "title":"Data"
          }
        },
        "type":"object",
        "required":[
          "code",
          "status",
          "message"
        ],
        "title":"APIResponse"
      },
      "Body_upload_yuanyanAgent_upload__post":{
        "properties":{
          "file":{
            "type":"string",
            "format":"binary",
            "title":"File"
          },
          "dataset_ids":{
            "type":"string",
            "title":"Dataset Ids",
            "description":"知识库 ID"
          }
        },
        "type":"object",
        "required":[
          "file",
          "dataset_ids"
        ],
        "title":"Body_upload_yuanyanAgent_upload__post"
      },
      "ChatRequest":{
        "properties":{
          "prompt":{
            "type":"string",
            "title":"Prompt",
            "description":"用户输入的提示词"
          },
          "dataset_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "minItems":0,
            "title":"Dataset Ids",
            "description":"知识库 ID 列表"
          }
        },
        "type":"object",
        "required":[
          "prompt",
          "dataset_ids"
        ],
        "title":"ChatRequest"
      },
      "HTTPValidationError":{
        "properties":{
          "detail":{
            "items":{
              "$ref":"#/components/schemas/ValidationError"
            },
            "type":"array",
            "title":"Detail"
          }
        },
        "type":"object",
        "title":"HTTPValidationError"
      },
      "ValidationError":{
        "properties":{
          "loc":{
            "items":{
              "anyOf":[
                {
                  "type":"string"
                },
                {
                  "type":"integer"
                }
              ]
            },
            "type":"array",
            "title":"Location"
          },
          "msg":{
            "type":"string",
            "title":"Message"
          },
          "type":{
            "type":"string",
            "title":"Error Type"
          }
        },
        "type":"object",
        "required":[
          "loc",
          "msg",
          "type"
        ],
        "title":"ValidationError"
      },
      "createProjectRequest":{
        "properties":{
          "dataset_name":{
            "type":"string",
            "title":"Dataset Name",
            "description":"知识库名称"
          }
        },
        "type":"object",
        "required":[
          "dataset_name"
        ],
        "title":"createProjectRequest"
      },
      "dagangGenrateRequest":{
        "properties":{
          "gn_name":{
            "type":"string",
            "title":"Gn Name",
            "description":"功能名称"
          },
          "gn_description":{
            "type":"string",
            "title":"Gn Description",
            "description":"上一步提取的功能需求规格说明"
          }
        },
        "type":"object",
        "required":[
          "gn_name",
          "gn_description"
        ],
        "title":"dagangGenrateRequest"
      },
      "deleteDocumentRequest":{
        "properties":{
          "dataset_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "minItems":1,
            "title":"Dataset Ids",
            "description":"该参数RAGFLOW的知识库ID（对应配置项或系统的ID）"
          },
          "doc_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "minItems":1,
            "title":"Doc Ids",
            "description":"该参数文档ID（对应文档ID）"
          }
        },
        "type":"object",
        "required":[
          "dataset_ids",
          "doc_ids"
        ],
        "title":"deleteDocumentRequest"
      },
      "deleteProjectRequest":{
        "properties":{
          "dataset_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "minItems":1,
            "title":"Dataset Ids",
            "description":"该参数RAGFLOW的知识库ID（对应配置项或系统的ID）"
          }
        },
        "type":"object",
        "required":[
          "dataset_ids"
        ],
        "title":"deleteProjectRequest"
      },
      "extractGNRequest":{
        "properties":{
          "dataset_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "title":"Dataset Ids",
            "description":"RAGFLOW的知识库ID（对应配置项或系统的ID）"
          },
          "doc_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "title":"Doc Ids",
            "description":"需求文档的ID"
          }
        },
        "type":"object",
        "required":[
          "dataset_ids",
          "doc_ids"
        ],
        "title":"extractGNRequest"
      },
      "extractXuqiuRequest":{
        "properties":{
          "dataset_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "title":"Dataset Ids",
            "description":"RAGFLOW的知识库ID（对应配置项或系统的ID）"
          },
          "doc_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "title":"Doc Ids",
            "description":"需求文档的ID"
          },
          "gn_name":{
            "type":"string",
            "title":"Gn Name",
            "description":"功能名称"
          }
        },
        "type":"object",
        "required":[
          "dataset_ids",
          "doc_ids",
          "gn_name"
        ],
        "title":"extractXuqiuRequest"
      },
      "listChunksRequest":{
        "properties":{
          "dataset_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "minItems":1,
            "title":"Dataset Ids",
            "description":"该参数RAGFLOW的知识库ID（对应配置项或系统的ID）"
          },
          "doc_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "minItems":1,
            "title":"Doc Ids",
            "description":"该参数文档ID（对应文档ID）"
          }
        },
        "type":"object",
        "required":[
          "dataset_ids",
          "doc_ids"
        ],
        "title":"listChunksRequest"
      },
      "optimizeXuqiuRequest":{
        "properties":{
          "dataset_ids":{
            "items":{
              "type":"string"
            },
            "type":"array",
            "minItems":1,
            "title":"Dataset Ids",
            "description":"该参数RAGFLOW的知识库ID（对应配置项或系统的ID）"
          },
          "gn_name":{
            "type":"string",
            "title":"Gn Name",
            "description":"功能名称"
          },
          "gn_description":{
            "type":"string",
            "title":"Gn Description",
            "description":"上一步提取的功能需求规格说明"
          },
          "wendang_error":{
            "type":"string",
            "title":"Wendang Error",
            "description":"文档审查问题"
          }
        },
        "type":"object",
        "required":[
          "dataset_ids",
          "gn_name",
          "gn_description",
          "wendang_error"
        ],
        "title":"optimizeXuqiuRequest"
      },
      "wendangshenchaRequest":{
        "properties":{
          "gn_name":{
            "type":"string",
            "title":"Gn Name",
            "description":"功能名称"
          },
          "gn_description":{
            "type":"string",
            "title":"Gn Description",
            "description":"上一步提取的功能需求规格说明"
          }
        },
        "type":"object",
        "required":[
          "gn_name",
          "gn_description"
        ],
        "title":"wendangshenchaRequest"
      },
      "yongliGenrateRequest":{
        "properties":{
          "dagang_description":{
            "type":"string",
            "title":"Dagang Description",
            "description":"上一步提取的测试大纲"
          }
        },
        "type":"object",
        "required":[
          "dagang_description"
        ],
        "title":"yongliGenrateRequest"
      }
    }
  }
}
```

