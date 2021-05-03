package com.amplifyframework.datastore.generated.model;


import androidx.core.util.ObjectsCompat;

import java.util.Objects;
import java.util.List;

/** This is an auto generated class representing the SongFeatures type in your schema. */
public final class SongFeatures {
  private final String song_id;
  private final Double danceability;
  private final Double energy;
  private final Double loudness;
  private final Double speechiness;
  private final Double acousticness;
  private final Double instrumentalness;
  private final Double liveness;
  private final Double valence;
  private final Double tempo;
  private final Song song;
  public String getSongId() {
      return song_id;
  }
  
  public Double getDanceability() {
      return danceability;
  }
  
  public Double getEnergy() {
      return energy;
  }
  
  public Double getLoudness() {
      return loudness;
  }
  
  public Double getSpeechiness() {
      return speechiness;
  }
  
  public Double getAcousticness() {
      return acousticness;
  }
  
  public Double getInstrumentalness() {
      return instrumentalness;
  }
  
  public Double getLiveness() {
      return liveness;
  }
  
  public Double getValence() {
      return valence;
  }
  
  public Double getTempo() {
      return tempo;
  }
  
  public Song getSong() {
      return song;
  }
  
  private SongFeatures(String song_id, Double danceability, Double energy, Double loudness, Double speechiness, Double acousticness, Double instrumentalness, Double liveness, Double valence, Double tempo, Song song) {
    this.song_id = song_id;
    this.danceability = danceability;
    this.energy = energy;
    this.loudness = loudness;
    this.speechiness = speechiness;
    this.acousticness = acousticness;
    this.instrumentalness = instrumentalness;
    this.liveness = liveness;
    this.valence = valence;
    this.tempo = tempo;
    this.song = song;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      SongFeatures songFeatures = (SongFeatures) obj;
      return ObjectsCompat.equals(getSongId(), songFeatures.getSongId()) &&
              ObjectsCompat.equals(getDanceability(), songFeatures.getDanceability()) &&
              ObjectsCompat.equals(getEnergy(), songFeatures.getEnergy()) &&
              ObjectsCompat.equals(getLoudness(), songFeatures.getLoudness()) &&
              ObjectsCompat.equals(getSpeechiness(), songFeatures.getSpeechiness()) &&
              ObjectsCompat.equals(getAcousticness(), songFeatures.getAcousticness()) &&
              ObjectsCompat.equals(getInstrumentalness(), songFeatures.getInstrumentalness()) &&
              ObjectsCompat.equals(getLiveness(), songFeatures.getLiveness()) &&
              ObjectsCompat.equals(getValence(), songFeatures.getValence()) &&
              ObjectsCompat.equals(getTempo(), songFeatures.getTempo()) &&
              ObjectsCompat.equals(getSong(), songFeatures.getSong());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getSongId())
      .append(getDanceability())
      .append(getEnergy())
      .append(getLoudness())
      .append(getSpeechiness())
      .append(getAcousticness())
      .append(getInstrumentalness())
      .append(getLiveness())
      .append(getValence())
      .append(getTempo())
      .append(getSong())
      .toString()
      .hashCode();
  }
  
  public static SongIdStep builder() {
      return new Builder();
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(song_id,
      danceability,
      energy,
      loudness,
      speechiness,
      acousticness,
      instrumentalness,
      liveness,
      valence,
      tempo,
      song);
  }
  public interface SongIdStep {
    DanceabilityStep songId(String songId);
  }
  

  public interface DanceabilityStep {
    EnergyStep danceability(Double danceability);
  }
  

  public interface EnergyStep {
    LoudnessStep energy(Double energy);
  }
  

  public interface LoudnessStep {
    SpeechinessStep loudness(Double loudness);
  }
  

  public interface SpeechinessStep {
    AcousticnessStep speechiness(Double speechiness);
  }
  

  public interface AcousticnessStep {
    InstrumentalnessStep acousticness(Double acousticness);
  }
  

  public interface InstrumentalnessStep {
    LivenessStep instrumentalness(Double instrumentalness);
  }
  

  public interface LivenessStep {
    ValenceStep liveness(Double liveness);
  }
  

  public interface ValenceStep {
    TempoStep valence(Double valence);
  }
  

  public interface TempoStep {
    SongStep tempo(Double tempo);
  }
  

  public interface SongStep {
    BuildStep song(Song song);
  }
  

  public interface BuildStep {
    SongFeatures build();
  }
  

  public static class Builder implements SongIdStep, DanceabilityStep, EnergyStep, LoudnessStep, SpeechinessStep, AcousticnessStep, InstrumentalnessStep, LivenessStep, ValenceStep, TempoStep, SongStep, BuildStep {
    private String song_id;
    private Double danceability;
    private Double energy;
    private Double loudness;
    private Double speechiness;
    private Double acousticness;
    private Double instrumentalness;
    private Double liveness;
    private Double valence;
    private Double tempo;
    private Song song;
    @Override
     public SongFeatures build() {
        
        return new SongFeatures(
          song_id,
          danceability,
          energy,
          loudness,
          speechiness,
          acousticness,
          instrumentalness,
          liveness,
          valence,
          tempo,
          song);
    }
    
    @Override
     public DanceabilityStep songId(String songId) {
        Objects.requireNonNull(songId);
        this.song_id = songId;
        return this;
    }
    
    @Override
     public EnergyStep danceability(Double danceability) {
        Objects.requireNonNull(danceability);
        this.danceability = danceability;
        return this;
    }
    
    @Override
     public LoudnessStep energy(Double energy) {
        Objects.requireNonNull(energy);
        this.energy = energy;
        return this;
    }
    
    @Override
     public SpeechinessStep loudness(Double loudness) {
        Objects.requireNonNull(loudness);
        this.loudness = loudness;
        return this;
    }
    
    @Override
     public AcousticnessStep speechiness(Double speechiness) {
        Objects.requireNonNull(speechiness);
        this.speechiness = speechiness;
        return this;
    }
    
    @Override
     public InstrumentalnessStep acousticness(Double acousticness) {
        Objects.requireNonNull(acousticness);
        this.acousticness = acousticness;
        return this;
    }
    
    @Override
     public LivenessStep instrumentalness(Double instrumentalness) {
        Objects.requireNonNull(instrumentalness);
        this.instrumentalness = instrumentalness;
        return this;
    }
    
    @Override
     public ValenceStep liveness(Double liveness) {
        Objects.requireNonNull(liveness);
        this.liveness = liveness;
        return this;
    }
    
    @Override
     public TempoStep valence(Double valence) {
        Objects.requireNonNull(valence);
        this.valence = valence;
        return this;
    }
    
    @Override
     public SongStep tempo(Double tempo) {
        Objects.requireNonNull(tempo);
        this.tempo = tempo;
        return this;
    }
    
    @Override
     public BuildStep song(Song song) {
        Objects.requireNonNull(song);
        this.song = song;
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String songId, Double danceability, Double energy, Double loudness, Double speechiness, Double acousticness, Double instrumentalness, Double liveness, Double valence, Double tempo, Song song) {
      super.songId(songId)
        .danceability(danceability)
        .energy(energy)
        .loudness(loudness)
        .speechiness(speechiness)
        .acousticness(acousticness)
        .instrumentalness(instrumentalness)
        .liveness(liveness)
        .valence(valence)
        .tempo(tempo)
        .song(song);
    }
    
    @Override
     public CopyOfBuilder songId(String songId) {
      return (CopyOfBuilder) super.songId(songId);
    }
    
    @Override
     public CopyOfBuilder danceability(Double danceability) {
      return (CopyOfBuilder) super.danceability(danceability);
    }
    
    @Override
     public CopyOfBuilder energy(Double energy) {
      return (CopyOfBuilder) super.energy(energy);
    }
    
    @Override
     public CopyOfBuilder loudness(Double loudness) {
      return (CopyOfBuilder) super.loudness(loudness);
    }
    
    @Override
     public CopyOfBuilder speechiness(Double speechiness) {
      return (CopyOfBuilder) super.speechiness(speechiness);
    }
    
    @Override
     public CopyOfBuilder acousticness(Double acousticness) {
      return (CopyOfBuilder) super.acousticness(acousticness);
    }
    
    @Override
     public CopyOfBuilder instrumentalness(Double instrumentalness) {
      return (CopyOfBuilder) super.instrumentalness(instrumentalness);
    }
    
    @Override
     public CopyOfBuilder liveness(Double liveness) {
      return (CopyOfBuilder) super.liveness(liveness);
    }
    
    @Override
     public CopyOfBuilder valence(Double valence) {
      return (CopyOfBuilder) super.valence(valence);
    }
    
    @Override
     public CopyOfBuilder tempo(Double tempo) {
      return (CopyOfBuilder) super.tempo(tempo);
    }
    
    @Override
     public CopyOfBuilder song(Song song) {
      return (CopyOfBuilder) super.song(song);
    }
  }
  
}
