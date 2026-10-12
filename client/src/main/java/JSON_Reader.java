import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.Paths;
import java.time.LocalDate;
import java.util.*;

public class JSON_Reader {

    public final List<Map<String, Object>> data = new ArrayList<>();
    private final ObjectMapper mapper = new ObjectMapper();

    @Deprecated
    public void addVideo(String title, String category) {
        addVideo(null, title, null, category, null, -1, -1);
    }

    public void addVideo(String videoId, String title,
                         String categoryId, String categoryName,
                         LocalDate likedAt, int durationSeconds, int weight) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("video_id", videoId);
        item.put("title", title);
        item.put("category_id", categoryId);
        item.put("category", categoryName);
        item.put("liked_at", likedAt == null ? null : likedAt.toString());
        item.put("duration_seconds", durationSeconds);
        item.put("weight", weight);
        data.add(item);
    }

    public void saveToJson(String filePath) throws Exception {
        mapper.writerWithDefaultPrettyPrinter()
                .writeValue(Paths.get(filePath).toFile(), data);
        System.out.println("Saved " + data.size() + " items to " + filePath);
    }
}