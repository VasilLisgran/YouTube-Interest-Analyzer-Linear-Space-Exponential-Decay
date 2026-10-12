import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.api.services.youtube.YouTube;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDate;
import java.util.*;


public class Main {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String PYTHON_DIR = "code";
    private static final String PYTHON_SCRIPT = "clusters.py";

    public static void main(String[] args) {
        try {
            Map<String, String> opt = parseArgs(args);

            String userId = opt.getOrDefault("user", "u8");
            String outDir = opt.getOrDefault("out", "data");
            double lambda = Double.parseDouble(
                    opt.getOrDefault("lambda", String.valueOf(User.DEFAULT_LAMBDA)));
            int maxDays = Integer.parseInt(
                    opt.getOrDefault("max-days", String.valueOf(User.DEFAULT_MAX_DAYS)));
            int topN = Integer.parseInt(opt.getOrDefault("top", "3"));
            int maxEvents = Integer.parseInt(opt.getOrDefault("max-events", "0"));

            int maxSearches = Integer.parseInt(opt.getOrDefault("max-searches", "20"));
            YouTubeDataLoader.WeightScheme weight = YouTubeDataLoader.WeightScheme
                    .valueOf(opt.getOrDefault("weight", "ORIGINAL"));

            LocalDate referenceDate = opt.containsKey("reference-date")
                    ? LocalDate.parse(opt.get("reference-date"))
                    : LocalDate.now();

            Files.createDirectories(Paths.get(outDir));
            String videosPath   = outDir + "/user_videos_"     + userId + ".json";
            String clustersPath = outDir + "/clusters_result_" + userId + ".json";
            String manifestPath = outDir + "/run_manifest_"    + userId + ".json";

            System.out.println("YouTube interest profiler — run " + userId);
            System.out.println("=================================================");

            YouTube youtube = YouTubeAuth.authenticate();
            CategoryRegistry registry = new CategoryRegistry();
            YouTubeDataLoader loader = new YouTubeDataLoader(youtube, registry, weight);

            System.out.println("category registry loaded (" + registry.getDimension() + " categories)");

            List<Event> events = loader.fetchLikedVideos(maxEvents, videosPath);

            User user = new User(userId, registry, referenceDate, lambda, maxDays);
            for (Event e : events) user.addEvent(e);
            user.calculateWithDecayAndDynamics();
            user.printUserVector();

            System.out.println("\n[IPC] running LaBSE/DBSCAN clustering...");
            if (!runPythonClustering(videosPath, clustersPath)) {
                System.err.println("clustering failed; aborting");
                return;
            }

            Map<String, Map<String, List<String>>> clusters = loadClusters(clustersPath);
            if (clusters == null || clusters.isEmpty()) {
                System.err.println("cluster file empty or malformed");
                return;
            }

            loader.generateRecommendations(user.getTopCategories(topN), clusters, maxSearches);

            String recsPath = outDir + "/recommendations_" + userId + ".json";
            MAPPER.writerWithDefaultPrettyPrinter()
                    .writeValue(new File(recsPath), loader.getRecommendations());

            writeManifest(manifestPath, userId, referenceDate, lambda, maxDays,
                    weight.name(), topN, events.size(), clusters,
                    maxSearches, loader.getSearchesUsed(),
                    loader.isQuotaExhausted());

            System.out.println("\nmanifest:        " + manifestPath);
            System.out.println("recommendations: " + recsPath);

        } catch (Exception e) {
            System.err.println("\nfatal: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> opt = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--")) continue;
            String key = args[i].substring(2);
            String val = (i + 1 < args.length && !args[i + 1].startsWith("--"))
                    ? args[++i] : "true";
            opt.put(key, val);
        }
        return opt;
    }

    private static void writeManifest(String path, String userId, LocalDate refDate,
                                      double lambda, int maxDays, String weightScheme,
                                      int topN, int nEvents,
                                      Map<String, Map<String, List<String>>> clusters,
                                      int searchBudget, int searchesUsed,
                                      boolean quotaExhausted)
            throws Exception {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("user_id", userId);
        m.put("reference_date", refDate.toString());
        m.put("lambda", lambda);
        m.put("half_life_days", Math.log(2.0) / -Math.log(lambda));
        m.put("max_days", maxDays);
        m.put("weight_scheme", weightScheme);
        m.put("top_n_categories", topN);
        m.put("n_events", nEvents);
        m.put("n_categories_clustered", clusters.size());
        int nClusters = clusters.values().stream().mapToInt(Map::size).sum();
        m.put("n_clusters_total", nClusters);
        m.put("search_budget", searchBudget);
        m.put("searches_used", searchesUsed);
        m.put("quota_exhausted", quotaExhausted);
        m.put("generated_at", java.time.Instant.now().toString());
        MAPPER.writerWithDefaultPrettyPrinter().writeValue(new File(path), m);
    }

    private static String resolvePython() {
        String fromEnv = System.getenv("PYTHON");
        if (fromEnv != null && !fromEnv.isBlank()) return fromEnv;
        for (String candidate : new String[]{"python3", "python"}) {
            try {
                Process p = new ProcessBuilder(candidate, "--version")
                        .redirectErrorStream(true).start();
                if (p.waitFor() == 0) return candidate;
            } catch (Exception ignored) {
                // try the next candidate
            }
        }
        return "python3";
    }

    private static boolean runPythonClustering(String videosPath, String clustersPath) {
        try {
            File workingDir = new File(PYTHON_DIR);
            if (!workingDir.isDirectory()) {
                System.err.println("not found: " + workingDir.getAbsolutePath());
                return false;
            }

            String python = resolvePython();
            ProcessBuilder pb = new ProcessBuilder(
                    python, PYTHON_SCRIPT,
                    "--input",  new File(videosPath).getAbsolutePath(),
                    "--output", new File(clustersPath).getAbsolutePath());
            pb.directory(workingDir);
            pb.redirectErrorStream(true);

            Process process = pb.start();
            try (BufferedReader r = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = r.readLine()) != null) {
                    System.out.println("   [python] " + line);
                }
            }
            return process.waitFor() == 0;

        } catch (Exception e) {
            System.err.println("   python error: " + e.getMessage());
            return false;
        }
    }

    public static Map<String, Map<String, List<String>>> loadClusters(String filePath) {
        try {
            Path path = Paths.get(filePath);
            if (!Files.exists(path)) return null;
            return MAPPER.readValue(path.toFile(), new TypeReference<>() {});
        } catch (Exception e) {
            System.err.println("   cluster load error: " + e.getMessage());
            return null;
        }
    }
}