package com.amplifyframework.datastore.generated.model;

import com.amplifyframework.core.model.temporal.Temporal;

import androidx.core.util.ObjectsCompat;

import java.util.Objects;
import java.util.List;

/** This is an auto generated class representing the Event type in your schema. */
public final class Event {
  private final Song song;
  private final Temporal.DateTime timestamp;
  private final Integer rating;
  public Song getSong() {
      return song;
  }
  
  public Temporal.DateTime getTimestamp() {
      return timestamp;
  }
  
  public Integer getRating() {
      return rating;
  }
  
  private Event(Song song, Temporal.DateTime timestamp, Integer rating) {
    this.song = song;
    this.timestamp = timestamp;
    this.rating = rating;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Event event = (Event) obj;
      return ObjectsCompat.equals(getSong(), event.getSong()) &&
              ObjectsCompat.equals(getTimestamp(), event.getTimestamp()) &&
              ObjectsCompat.equals(getRating(), event.getRating());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getSong())
      .append(getTimestamp())
      .append(getRating())
      .toString()
      .hashCode();
  }
  
  public static SongStep builder() {
      return new Builder();
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(song,
      timestamp,
      rating);
  }
  public interface SongStep {
    BuildStep song(Song song);
  }
  

  public interface BuildStep {
    Event build();
    BuildStep timestamp(Temporal.DateTime timestamp);
    BuildStep rating(Integer rating);
  }
  

  public static class Builder implements SongStep, BuildStep {
    private Song song;
    private Temporal.DateTime timestamp;
    private Integer rating;
    @Override
     public Event build() {
        
        return new Event(
          song,
          timestamp,
          rating);
    }
    
    @Override
     public BuildStep song(Song song) {
        Objects.requireNonNull(song);
        this.song = song;
        return this;
    }
    
    @Override
     public BuildStep timestamp(Temporal.DateTime timestamp) {
        this.timestamp = timestamp;
        return this;
    }
    
    @Override
     public BuildStep rating(Integer rating) {
        this.rating = rating;
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(Song song, Temporal.DateTime timestamp, Integer rating) {
      super.song(song)
        .timestamp(timestamp)
        .rating(rating);
    }
    
    @Override
     public CopyOfBuilder song(Song song) {
      return (CopyOfBuilder) super.song(song);
    }
    
    @Override
     public CopyOfBuilder timestamp(Temporal.DateTime timestamp) {
      return (CopyOfBuilder) super.timestamp(timestamp);
    }
    
    @Override
     public CopyOfBuilder rating(Integer rating) {
      return (CopyOfBuilder) super.rating(rating);
    }
  }
  
}
