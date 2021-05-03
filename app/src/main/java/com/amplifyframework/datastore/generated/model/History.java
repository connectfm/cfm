package com.amplifyframework.datastore.generated.model;


import androidx.core.util.ObjectsCompat;

import java.util.Objects;
import java.util.List;

/** This is an auto generated class representing the History type in your schema. */
public final class History {
  private final String lastUpdated;
  private final List<Event> events;
  public String getLastUpdated() {
      return lastUpdated;
  }
  
  public List<Event> getEvents() {
      return events;
  }
  
  private History(String lastUpdated, List<Event> events) {
    this.lastUpdated = lastUpdated;
    this.events = events;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      History history = (History) obj;
      return ObjectsCompat.equals(getLastUpdated(), history.getLastUpdated()) &&
              ObjectsCompat.equals(getEvents(), history.getEvents());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getLastUpdated())
      .append(getEvents())
      .toString()
      .hashCode();
  }
  
  public static BuildStep builder() {
      return new Builder();
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(lastUpdated,
      events);
  }
  public interface BuildStep {
    History build();
    BuildStep lastUpdated(String lastUpdated);
    BuildStep events(List<Event> events);
  }
  

  public static class Builder implements BuildStep {
    private String lastUpdated;
    private List<Event> events;
    @Override
     public History build() {
        
        return new History(
          lastUpdated,
          events);
    }
    
    @Override
     public BuildStep lastUpdated(String lastUpdated) {
        this.lastUpdated = lastUpdated;
        return this;
    }
    
    @Override
     public BuildStep events(List<Event> events) {
        this.events = events;
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String lastUpdated, List<Event> events) {
      super.lastUpdated(lastUpdated)
        .events(events);
    }
    
    @Override
     public CopyOfBuilder lastUpdated(String lastUpdated) {
      return (CopyOfBuilder) super.lastUpdated(lastUpdated);
    }
    
    @Override
     public CopyOfBuilder events(List<Event> events) {
      return (CopyOfBuilder) super.events(events);
    }
  }
  
}
