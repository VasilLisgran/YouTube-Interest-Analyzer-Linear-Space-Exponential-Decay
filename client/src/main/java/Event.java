import java.time.LocalDate;

// User's events : what did user do?
public class Event {
    private LocalDate date;
    private String categoryId;
    private int watchTime;

    public Event() { }
    public Event(LocalDate date, String categoryId, int watchTime){
        this.date = date;
        this.categoryId = categoryId;
        this.watchTime = watchTime;
    }

    public LocalDate getDate() { return date; }
    public String getCategoryId() { return categoryId; }
    public int getWatchTime() { return watchTime; }

}