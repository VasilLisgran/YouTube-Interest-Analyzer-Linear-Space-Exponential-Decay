package recommender.Api;

import com.google.api.services.youtube.YouTube;
import com.google.api.services.youtube.model.*;
import recommender.JSON_Reader;
import recommender.Model.CategoryRegistry;
import recommender.Model.Event;

import java.io.IOException;
import java.time.Duration;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Loading API YouTube
 * Optimization & filtering
 */
public class YouTubeDataLoader {
    private final JSON_Reader jsonReader = new JSON_Reader();
    private final YouTube youtube;
    public final CategoryRegistry categoryRegistry;

    // Liked videos
    private final Set<String> watchedTitles = new HashSet<>();

    public YouTubeDataLoader(YouTube youtube, CategoryRegistry categoryRegistry) {
        this.youtube = youtube;
        this.categoryRegistry = categoryRegistry;
    }

    public List<Event> fetchLikedVideos(int maxEvents) throws Exception {
        List<Event> events = new ArrayList<>();
        String pageToken = null;
        DateTimeFormatter formatter = DateTimeFormatter.ISO_DATE_TIME;

        System.out.println("Loading likes (API)");
        int totalFetched = 0;

        do {
            // 1. Getting likes
            YouTube.PlaylistItems.List playlistRequest = youtube.playlistItems()
                    .list(Arrays.asList("snippet", "contentDetails"));
            playlistRequest.setPlaylistId("LL");
            playlistRequest.setMaxResults(50L);
            if (pageToken != null) {
                playlistRequest.setPageToken(pageToken);
            }

            PlaylistItemListResponse playlistResponse = playlistRequest.execute();
            List<PlaylistItem> items = playlistResponse.getItems();

            if (items == null || items.isEmpty()) break;

            // 2. Getting id of full page
            List<String> videoIds = items.stream()
                    .map(item -> item.getContentDetails().getVideoId())
                    .collect(Collectors.toList());

            Map<String, VideoDetails> detailsMap = fetchVideoDetailsBatch(videoIds);

            // 3. Results
            for (PlaylistItem item : items) {
                String videoId = item.getContentDetails().getVideoId();
                String title = item.getSnippet().getTitle();
                String publishedAtStr = item.getSnippet().getPublishedAt().toString();

                LocalDate likedDate = LocalDate.parse(publishedAtStr, formatter);

                // Default parameters
                VideoDetails details = detailsMap.getOrDefault(videoId, new VideoDetails("22", "PT3M0S"));
                String categoryId = details.getCategoryId();

                // Antipam shorts
                long durationSec = parseISO8601Duration(details.getDurationISO());
                int calculatedWeight = calculateAdaptiveWeight(durationSec);

                Event event = new Event(likedDate, categoryId, calculatedWeight);
                events.add(event);

                // Saving watched
                watchedTitles.add(title.toLowerCase().trim());

                String categoryName = categoryRegistry.getCategoryName(categoryId);
                if (categoryName == null) categoryName = "People & Blogs";

                jsonReader.addVideo(title, categoryName);

                totalFetched++;
                if (maxEvents > 0 && totalFetched >= maxEvents) break;
            }

            if (maxEvents > 0 && totalFetched >= maxEvents) break;
            pageToken = playlistResponse.getNextPageToken();

        } while (pageToken != null);

        jsonReader.saveToJson("../data/user_videos.json");
        System.out.printf(Locale.US, "📊 [API] Успешно обработано событий: %d%n", events.size());
        return events;
    }


    private Map<String, VideoDetails> fetchVideoDetailsBatch(List<String> videoIds) throws IOException {
        Map<String, VideoDetails> map = new HashMap<>();
        if (videoIds == null || videoIds.isEmpty()) return map;

        YouTube.Videos.List videoRequest = youtube.videos().list(Arrays.asList("snippet", "contentDetails"));
        videoRequest.setId(videoIds);

        VideoListResponse response = videoRequest.execute();
        List<Video> videos = response.getItems();

        if (videos != null) {
            for (Video v : videos) {
                String catId = v.getSnippet().getCategoryId();
                String duration = v.getContentDetails().getDuration();
                map.put(v.getId(), new VideoDetails(catId, duration));
            }
        }
        return map;
    }


    private int calculateAdaptiveWeight(long durationSeconds) {
        if (durationSeconds <= 60) {
            return 10; // Fixed for shorts
        }
        // For long videos
        return (int) (40 + Math.log(durationSeconds) * 15);
    }

    private long parseISO8601Duration(String isoDuration) {
        if (isoDuration == null || isoDuration.isEmpty()) return 180;
        try {
            return Duration.parse(isoDuration).getSeconds();
        } catch (Exception e) {
            return 180;
        }
    }

    public void generateRecommendations(List<Map.Entry<String, Double>> topCategories, Map<String, Map<String, List<String>>> clusters) throws IOException {
        System.out.println("\n Recommendations:");
        System.out.println("=========================================");

        for (Map.Entry<String, Double> entry : topCategories) {
            String category = entry.getKey();
            double cosineScore = entry.getValue();

            if (cosineScore < 0.05) continue; // Cut cold categories

            Map<String, List<String>> catClusters = clusters.get(category);
            if (catClusters == null || catClusters.isEmpty()) continue;

            System.out.printf(Locale.US, "%n📂 Category [%s] (Relevance: %.4f):%n", category, cosineScore);

            int totalToShow = Math.max(3, (int) (cosineScore * 50));
            int perClusterLimit = Math.max(2, totalToShow / catClusters.size());

            for (var cluster : catClusters.entrySet()) {
                String query = String.join(" ", cluster.getValue());
                System.out.printf("  🔎 Searching by cluster: \"%s\"%n", query);
                searchAndFilterOnYouTube(query, perClusterLimit, category);
            }
        }
    }

    /**
     * Cut duplicates
     */
    private void searchAndFilterOnYouTube(String query, int limit, String category) throws IOException {
        YouTube.Search.List request = youtube.search().list(List.of("snippet"));
        request.setQ(query);
        request.setType(List.of("video"));
        request.setMaxResults(25L);
        request.setOrder(category.equals("Gaming") ? "date" : "relevance"); // only new gaming videos

        SearchListResponse response = request.execute();
        List<SearchResult> results = response.getItems();

        if (results == null || results.isEmpty()) {
            System.out.println("   ⚠️ Not found");
            return;
        }

        int displayed = 0;
        for (SearchResult result : results) {
            String title = result.getSnippet().getTitle();

            // Don't show liked videos
            if (watchedTitles.contains(title.toLowerCase().trim())) {
                continue;
            }

            System.out.printf("   %d. %s%n", ++displayed, title);
            System.out.printf("      https://youtube.com/watch?v=%s%n", result.getId().getVideoId());

            if (displayed >= limit) break;
        }

        if (displayed == 0) {
            System.out.println("   ⚠️ All searched videos are liked by user.");
        }
    }

    private static class VideoDetails {
        private final String categoryId;
        private final String durationISO;

        public VideoDetails(String categoryId, String durationISO) {
            this.categoryId = categoryId;
            this.durationISO = durationISO;
        }
        public String getCategoryId() { return categoryId; }
        public String getDurationISO() { return durationISO; }
    }
}
