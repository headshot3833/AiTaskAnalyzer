using Microsoft.ML.OnnxRuntimeGenAI;

namespace AiTaskAnalyzer.Services
{
    public class LocalOnnxAiService : IDisposable
    {
        private readonly Model _model;
        private readonly Tokenizer _tokenizer;
        private readonly object _lock = new();

        public LocalOnnxAiService(IWebHostEnvironment environment)
        {
            var modelPath = Path.Combine(environment.ContentRootPath, "Models", "phi3");

            if (!Directory.Exists(modelPath))
            {
                throw new DirectoryNotFoundException(
                    $"Папка с ONNX-моделью не найдена: {modelPath}");
            }

            _model = new Model(modelPath);
            _tokenizer = new Tokenizer(_model);
        }

        public string AnalyzeClientRequest(string clientText)
        {
            var prompt = BuildPrompt(clientText);

            lock (_lock)
            {
                using var sequences = _tokenizer.Encode(prompt);

                using var generatorParams = new GeneratorParams(_model);

                generatorParams.SetSearchOption("max_length", 600);
                generatorParams.SetSearchOption("temperature", 0.1);
                generatorParams.SetSearchOption("top_p", 0.8);

                using var generator = new Generator(_model, generatorParams);

                generator.AppendTokenSequences(sequences);

                while (!generator.IsDone())
                {
                    generator.GenerateNextToken();
                }

                var output = _tokenizer.Decode(generator.GetSequence(0));

                return ClearModelOutput(output, prompt);
            }
        }

        private static string BuildPrompt(string clientText)
        {
            return $"""
                        <|system|>
                        Ты AI-классификатор обращений клиентов IT-компании DDPlanet.
                        Твоя задача — классифицировать обращение клиента.
                        Отвечай только на русском языке.
                        Не придумывай лишние детали.
                        Не составляй длинное техническое задание.
                        Не добавляй задачи, шаги, подзадачи и нумерацию внутри пунктов.
                        Ответ должен быть коротким.
                        <|end|>
                        <|user|>
                        Обращение клиента:
                        "{clientText}"

                        Верни только результат в таком формате:

                        1. Тип задачи: выбери одно из значений: баг, новая функциональность, консультация, поддержка, не определено
                        2. Приоритет: выбери одно из значений: низкий, средний, высокий
                        3. Ответственный отдел: выбери одно из значений: Backend, Frontend, QA, UX/UI, Support, Mobile
                        4. Краткое описание задачи: одно предложение
                        5. Черновик технического задания: максимум два предложения
                        <|end|>
                        <|assistant|>
                        """;
        }
        private static string ClearModelOutput(string output, string prompt)
        {
            if (string.IsNullOrWhiteSpace(output))
                return string.Empty;

            output = output
                .Replace("<|system|>", string.Empty)
                .Replace("<|user|>", string.Empty)
                .Replace("<|assistant|>", string.Empty)
                .Replace("<|end|>", string.Empty)
                .Trim();

            const string resultStart = "1. Тип задачи:";

            var index = output.LastIndexOf(resultStart, StringComparison.OrdinalIgnoreCase);

            if (index >= 0)
            {
                output = output[index..];
            }

            return output.Trim();
        }
        public void Dispose()
        {
            _tokenizer.Dispose();
            _model.Dispose();
        }
    }
}
