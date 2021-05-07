package com.amplifyframework.datastore.generated.model;

import com.amplifyframework.core.model.annotations.BelongsTo;
import com.amplifyframework.core.model.temporal.Temporal;

import java.util.List;
import java.util.UUID;
import java.util.Objects;

import androidx.core.util.ObjectsCompat;

import com.amplifyframework.core.model.AuthStrategy;
import com.amplifyframework.core.model.Model;
import com.amplifyframework.core.model.ModelOperation;
import com.amplifyframework.core.model.annotations.AuthRule;
import com.amplifyframework.core.model.annotations.Index;
import com.amplifyframework.core.model.annotations.ModelConfig;
import com.amplifyframework.core.model.annotations.ModelField;
import com.amplifyframework.core.model.query.predicate.QueryField;

import static com.amplifyframework.core.model.query.predicate.QueryField.field;

/** This is an auto generated class representing the Event type in your schema. */
@SuppressWarnings("all")
@ModelConfig(pluralName = "Events", authRules = {
  @AuthRule(allow = AuthStrategy.PUBLIC, operations = { ModelOperation.CREATE, ModelOperation.UPDATE, ModelOperation.DELETE, ModelOperation.READ })
})
public final class Event implements Model {
  public static final QueryField ID = field("id");
  public static final QueryField SONG = field("eventSongId");
  public static final QueryField TIMESTAMP = field("timestamp");
  public static final QueryField RATING = field("rating");
  public static final QueryField HISTORY_EVENTS_ID = field("historyEventsId");
  private final @ModelField(targetType="ID", isRequired = true) String id;
  private final @ModelField(targetType="Song", isRequired = true) @BelongsTo(targetName = "eventSongId", type = Song.class) Song song;
  private final @ModelField(targetType="AWSTimestamp") Temporal.Timestamp timestamp;
  private final @ModelField(targetType="Int") Integer rating;
  private final @ModelField(targetType="ID") String historyEventsId;
  public String getId() {
      return id;
  }
  
  public Song getSong() {
      return song;
  }
  
  public Temporal.Timestamp getTimestamp() {
      return timestamp;
  }
  
  public Integer getRating() {
      return rating;
  }
  
  public String getHistoryEventsId() {
      return historyEventsId;
  }
  
  private Event(String id, Song song, Temporal.Timestamp timestamp, Integer rating, String historyEventsId) {
    this.id = id;
    this.song = song;
    this.timestamp = timestamp;
    this.rating = rating;
    this.historyEventsId = historyEventsId;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Event event = (Event) obj;
      return ObjectsCompat.equals(getId(), event.getId()) &&
              ObjectsCompat.equals(getSong(), event.getSong()) &&
              ObjectsCompat.equals(getTimestamp(), event.getTimestamp()) &&
              ObjectsCompat.equals(getRating(), event.getRating()) &&
              ObjectsCompat.equals(getHistoryEventsId(), event.getHistoryEventsId());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getId())
      .append(getSong())
      .append(getTimestamp())
      .append(getRating())
      .append(getHistoryEventsId())
      .toString()
      .hashCode();
  }
  
  @Override
   public String toString() {
    return new StringBuilder()
      .append("Event {")
      .append("id=" + String.valueOf(getId()) + ", ")
      .append("song=" + String.valueOf(getSong()) + ", ")
      .append("timestamp=" + String.valueOf(getTimestamp()) + ", ")
      .append("rating=" + String.valueOf(getRating()) + ", ")
      .append("historyEventsId=" + String.valueOf(getHistoryEventsId()))
      .append("}")
      .toString();
  }
  
  public static SongStep builder() {
      return new Builder();
  }
  
  /** 
   * WARNING: This method should not be used to build an instance of this object for a CREATE mutation.
   * This is a convenience method to return an instance of the object with only its ID populated
   * to be used in the context of a parameter in a delete mutation or referencing a foreign key
   * in a relationship.
   * @param id the id of the existing item this instance will represent
   * @return an instance of this model with only ID populated
   * @throws IllegalArgumentException Checks that ID is in the proper format
   */
  public static Event justId(String id) {
    try {
      UUID.fromString(id); // Check that ID is in the UUID format - if not an exception is thrown
    } catch (Exception exception) {
      throw new IllegalArgumentException(
              "Model IDs must be unique in the format of UUID. This method is for creating instances " +
              "of an existing object with only its ID field for sending as a mutation parameter. When " +
              "creating a new object, use the standard builder method and leave the ID field blank."
      );
    }
    return new Event(
      id,
      null,
      null,
      null,
      null
    );
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(id,
      song,
      timestamp,
      rating,
      historyEventsId);
  }
  public interface SongStep {
    BuildStep song(Song song);
  }
  

  public interface BuildStep {
    Event build();
    BuildStep id(String id) throws IllegalArgumentException;
    BuildStep timestamp(Temporal.Timestamp timestamp);
    BuildStep rating(Integer rating);
    BuildStep historyEventsId(String historyEventsId);
  }
  

  public static class Builder implements SongStep, BuildStep {
    private String id;
    private Song song;
    private Temporal.Timestamp timestamp;
    private Integer rating;
    private String historyEventsId;
    @Override
     public Event build() {
        String id = this.id != null ? this.id : UUID.randomUUID().toString();
        
        return new Event(
          id,
          song,
          timestamp,
          rating,
          historyEventsId);
    }
    
    @Override
     public BuildStep song(Song song) {
        Objects.requireNonNull(song);
        this.song = song;
        return this;
    }
    
    @Override
     public BuildStep timestamp(Temporal.Timestamp timestamp) {
        this.timestamp = timestamp;
        return this;
    }
    
    @Override
     public BuildStep rating(Integer rating) {
        this.rating = rating;
        return this;
    }
    
    @Override
     public BuildStep historyEventsId(String historyEventsId) {
        this.historyEventsId = historyEventsId;
        return this;
    }
    
    /** 
     * WARNING: Do not set ID when creating a new object. Leave this blank and one will be auto generated for you.
     * This should only be set when referring to an already existing object.
     * @param id id
     * @return Current Builder instance, for fluent method chaining
     * @throws IllegalArgumentException Checks that ID is in the proper format
     */
    public BuildStep id(String id) throws IllegalArgumentException {
        this.id = id;
        
        try {
            UUID.fromString(id); // Check that ID is in the UUID format - if not an exception is thrown
        } catch (Exception exception) {
          throw new IllegalArgumentException("Model IDs must be unique in the format of UUID.",
                    exception);
        }
        
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String id, Song song, Temporal.Timestamp timestamp, Integer rating, String historyEventsId) {
      super.id(id);
      super.song(song)
        .timestamp(timestamp)
        .rating(rating)
        .historyEventsId(historyEventsId);
    }
    
    @Override
     public CopyOfBuilder song(Song song) {
      return (CopyOfBuilder) super.song(song);
    }
    
    @Override
     public CopyOfBuilder timestamp(Temporal.Timestamp timestamp) {
      return (CopyOfBuilder) super.timestamp(timestamp);
    }
    
    @Override
     public CopyOfBuilder rating(Integer rating) {
      return (CopyOfBuilder) super.rating(rating);
    }
    
    @Override
     public CopyOfBuilder historyEventsId(String historyEventsId) {
      return (CopyOfBuilder) super.historyEventsId(historyEventsId);
    }
  }
  
}
