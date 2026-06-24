package recommender;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.api.services.youtube.YouTube;
import recommender.Api.YouTubeAuth;
import recommender.Api.YouTubeDataLoader;
import recommender.Model.CategoryRegistry;
import recommender.Model.Event;
import recommender.Model.User;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

/**
 * Главный точка входа системы рекомендаций.
 * Демонстрирует паттерн оркестрации, управление внешними процессами (IPC) и обработку ошибок.
 */
public class Main {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String DATA_DIR = "../data";
    private static final String CLUSTERS_DIR = "../clusters";
    private static final String CLUSTERS_RESULT_PATH = DATA_DIR + "/clusters_result.json";
    private static final String PYTHON_SCRIPT = "clusters.py";

    public static void main(String[] args) {
        try {
            System.out.println("🎬 Рекомендательная система на базе YouTube API и NLP");
            System.out.println("=================================================");

            // Инициализация инфраструктурного слоя
            System.out.println("🔐 Авторизация в Google OAuth 2.0...");
            YouTube youtube = YouTubeAuth.authenticate();

            CategoryRegistry categoryRegistry = new CategoryRegistry();
            YouTubeDataLoader dataLoader = new YouTubeDataLoader(youtube, categoryRegistry);

            System.out.println("✅ Справочник категорий успешно загружен (" + categoryRegistry.getDimension() + " векторов)");

            // Загрузка данных пользователя
            User user = new User("TargetUser", categoryRegistry);
            List<Event> events = dataLoader.fetchLikedVideos(100); // Ограничим 100 для демонстрации

            for (Event event : events) {
                user.addEvent(event);
            }

            // Математический обсчет профиля
            user.calculateWithDecayAndDynamics();
            user.printUserVector();

            // Интеграция с Python-модулем машинного обучения (LaBSE Clustering)
            System.out.println("\n🤖 [IPC] Запуск конвейера NLP кластеризации на Python...");
            if (!runPythonClustering()) {
                System.err.println("❌ Ошибка выполнения Python скрипта кластеризации. Завершение работы.");
                return;
            }

            // Загрузка результатов кластеризации и генерация рекомендаций
            Map<String, Map<String, List<String>>> clusters = loadClustersFromJson(CLUSTERS_RESULT_PATH);
            if (clusters == null || clusters.isEmpty()) {
                System.err.println("⚠️ Данные кластеров пусты или повреждены.");
                return;
            }

            // Ранжирование топ-категорий на основе косинусного сходства
            var topCategories = user.getTopCategories(3);
            dataLoader.generateRecommendations(topCategories, clusters);

        } catch (Exception e) {
            System.err.println("\n❌ Критический сбой ядра приложения: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Безопасный запуск внешнего ML скрипта.
     * Исключает возникновение Deadlock за счет правильного вычитывания буфера потока.
     */
    /**
     * Безопасный и кроссплатформенный запуск внешнего ML скрипта.
     * Автоматически определяет команду (python/python3) и валидирует пути.
     */
    private static boolean runPythonClustering() {
        try {
            String pythonFolderName = "PythonSystem";
            String scriptName = "clusters.py";

            File workingDir = new File("../" + pythonFolderName);

            if (!workingDir.exists() || !workingDir.isDirectory()) {
                System.err.println("   ❌Error: not found path: " + workingDir.getAbsolutePath());
                return false;
            }

            // For Mac
            String pythonCommand = "python3";
            String[] standardMacPaths = {
                    "/opt/homebrew/bin/python3",
                    "/usr/local/bin/python3",
                    "/usr/bin/python3"
            };

            for (String path : standardMacPaths) {
                if (new File(path).exists()) {
                    pythonCommand = path;
                    break;
                }
            }


            ProcessBuilder pb = new ProcessBuilder(pythonCommand, scriptName);
            pb.directory(workingDir);
            pb.redirectErrorStream(true);

            Process process = pb.start();

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println("   [Python ML] " + line);
                }
            }

            int exitCode = process.waitFor();
            return exitCode == 0;

        } catch (Exception e) {
            System.out.println("   ⚠️ Python error: " + e.getMessage());
            return false;
        }
    }

    public static Map<String, Map<String, List<String>>> loadClustersFromJson(String filePath) {
        try {
            Path path = Paths.get(filePath);
            if (!Files.exists(path)) {
                return null;
            }
            return MAPPER.readValue(path.toFile(), new TypeReference<>() {});
        } catch (Exception e) {
            System.err.println("   ⚠️ Clusters error: " + e.getMessage());
            return null;
        }
    }
}
