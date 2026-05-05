using AiTaskAnalyzer.Models;

namespace AiTaskAnalyzer.Interfaces
{
    public interface IAiAnalyzeService
    {
        AnalyzeResponse Analyze(AnalyzeRequest request);
    }
}
